require 'nn'
require 'nngraph'
require 'hdf5'

require 'data.lua'
require 'models.lua'
require 'utils.lua'

cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/entail-train.hdf5', [[Path to the training *.hdf5 file]])
cmd:option('-val_data_file', 'data/entail-val.hdf5', [[Path to validation *.hdf5 file]])
cmd:option('-test_data_file','data/entail-test.hdf5',[[Path to test *.hdf5 file]])

cmd:option('-savefile', 'model', [[Savefile name]])

-- model specs
cmd:option('-hidden_size', 200, [[MLP hidden layer size]])
cmd:option('-word_vec_size', 300, [[Word embedding size]])
cmd:option('-share_params',1, [[Share parameters between the two sentence encoders]])
cmd:option('-dropout', 0.2, [[Dropout probability.]])   

-- optimization
cmd:option('-epochs', 100, [[Number of training epochs]])
cmd:option('-param_init', 0.01, [[Parameters are initialized over uniform distribution with support
                               (-param_init, param_init)]])
cmd:option('-optim', 'adagrad', [[Optimization method. Possible options are: 
                              sgd (vanilla SGD), adagrad, adadelta, adam]])
cmd:option('-learning_rate', 0.05, [[Starting learning rate. If adagrad/adadelta/adam is used, 
                                then this is the global learning rate.]])
cmd:option('-pre_word_vecs', 'glove.hdf5', [[If a valid path is specified, then this will load 
                                      pretrained word embeddings (hdf5 file)]])
cmd:option('-fix_word_vecs', 1, [[If = 1, fix word embeddings]])
cmd:option('-max_batch_l', '', [[If blank, then it will infer the max batch size from the
				   data.]])
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-print_every', 1000, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

function zero_table(t)
  for i = 1, #t do
    t[i]:zero()
  end
end

function train(train_data, valid_data)

  local timer = torch.Timer()
  local start_decay = 0
  params, grad_params = {}, {}
  opt.train_perf = {}
  opt.val_perf = {}
  
  for i = 1, #layers do
    local p, gp = layers[i]:getParameters()
    local rand_vec = torch.randn(p:size(1)):mul(opt.param_init)
    if opt.gpuid >= 0 then
      rand_vec = rand_vec:cuda()
    end	 
    p:copy(rand_vec)	 
    params[i] = p
    grad_params[i] = gp
  end
  if opt.pre_word_vecs:len() > 0 then
    print("loading pre-trained word vectors")
    local f = hdf5.open(opt.pre_word_vecs)     
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vecs_enc1.weight[i]:copy(pre_word_vecs[i])
      word_vecs_enc2.weight[i]:copy(pre_word_vecs[i])       
    end
  end

  --copy shared params   
  params[2]:copy(params[1])   
  if opt.share_params == 1 then
    all_layers.proj2.weight:copy(all_layers.proj1.weight)
    for k = 2, 5, 3 do	 
      all_layers.f2.modules[k].weight:copy(all_layers.f1.modules[k].weight)
      all_layers.f2.modules[k].bias:copy(all_layers.f1.modules[k].bias)
      all_layers.g2.modules[k].weight:copy(all_layers.g1.modules[k].weight)
      all_layers.g2.modules[k].bias:copy(all_layers.g1.modules[k].bias)
    end      
  end
  
  -- prototypes for gradients so there is no need to clone
  word_vecs1_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l_src, opt.word_vec_size)
  word_vecs2_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l_targ, opt.word_vec_size)
  
  if opt.gpuid >= 0 then
    cutorch.setDevice(opt.gpuid)                        
    word_vecs1_grad_proto = word_vecs1_grad_proto:cuda()
    word_vecs2_grad_proto = word_vecs2_grad_proto:cuda()
  end

  function train_batch(data, epoch)
    local train_loss = 0
    local train_sents = 0
    local batch_order = torch.randperm(data.length) -- shuffle mini batch order     
    local start_time = timer:time().real
    local num_words_target = 0
    local num_words_source = 0
    local train_num_correct = 0 
    sent_encoder:training()
    for i = 1, data:size() do
      zero_table(grad_params, 'zero')
      local d = data[batch_order[i]]
      local target, source, batch_l, target_l, source_l, label = table.unpack(d)	 
      
      -- resize the various temporary tensors that are going to hold contexts/grads
      local word_vecs1_grads = word_vecs1_grad_proto[{{1, batch_l}, {1, source_l}}]:zero()
      local word_vecs2_grads = word_vecs2_grad_proto[{{1, batch_l}, {1, target_l}}]:zero()
      local word_vecs1 = word_vecs_enc1:forward(source)
      local word_vecs2 = word_vecs_enc2:forward(target)	 
      set_size_encoder(batch_l, source_l, target_l,
		       opt.word_vec_size, opt.hidden_size, all_layers)
      local pred_input = {word_vecs1, word_vecs2}
      local pred_label = sent_encoder:forward(pred_input)
      local _, pred_argmax = pred_label:max(2)
      train_num_correct = train_num_correct + pred_argmax:double():view(batch_l):eq(label:double()):sum()	 
      local loss = disc_criterion:forward(pred_label, label)
      local dl_dp = disc_criterion:backward(pred_label, label)
      dl_dp:div(batch_l)
      local dl_dinput1, dl_dinput2 = table.unpack(sent_encoder:backward(pred_input, dl_dp))    
      word_vecs_enc1:backward(source, dl_dinput1)
      word_vecs_enc2:backward(target, dl_dinput2)
      
      if opt.fix_word_vecs == 1 then
	word_vecs_enc1.gradWeight:zero()
	word_vecs_enc2.gradWeight:zero()	   
      end
      
      grad_params[1]:add(grad_params[2])
      grad_params[2]:zero()

      if opt.share_params == 1 then
	all_layers.proj1.gradWeight:add(all_layers.proj2.gradWeight)
	all_layers.proj2.gradWeight:zero()
	for k = 2, 5, 3 do	       
	  all_layers.f1.modules[k].gradWeight:add(all_layers.f2.modules[k].gradWeight)
	  all_layers.f1.modules[k].gradBias:add(all_layers.f2.modules[k].gradBias)
	  all_layers.g1.modules[k].gradWeight:add(all_layers.g2.modules[k].gradWeight)
	  all_layers.g1.modules[k].gradBias:add(all_layers.g2.modules[k].gradBias)
	  all_layers.f2.modules[k].gradWeight:zero()
	  all_layers.f2.modules[k].gradBias:zero()
	  all_layers.g2.modules[k].gradWeight:zero()
	  all_layers.g2.modules[k].gradBias:zero()
	end	
      end	 
      
      -- Update params
      for j = 1, #grad_params do
	if opt.optim == 'adagrad' then
	  adagrad_step(params[j], grad_params[j], layer_etas[j], optStates[j])
	elseif opt.optim == 'adadelta' then
	  adadelta_step(params[j], grad_params[j], layer_etas[j], optStates[j])
	elseif opt.optim == 'adam' then
	  adam_step(params[j], grad_params[j], layer_etas[j], optStates[j])	       
	else
	  params[j]:add(grad_params[j]:mul(-opt.learning_rate))       
	end
      end	 

      params[2]:copy(params[1])
      if opt.share_params == 1 then
	all_layers.proj2.weight:copy(all_layers.proj1.weight)
	for k = 2, 5, 3 do	 
	  all_layers.f2.modules[k].weight:copy(all_layers.f1.modules[k].weight)
	  all_layers.f2.modules[k].bias:copy(all_layers.f1.modules[k].bias)
	  all_layers.g2.modules[k].weight:copy(all_layers.g1.modules[k].weight)
	  all_layers.g2.modules[k].bias:copy(all_layers.g1.modules[k].bias)
	end      
      end
      
      -- Bookkeeping
      num_words_target = num_words_target + batch_l*target_l
      num_words_source = num_words_source + batch_l*source_l
      train_loss = train_loss + loss
      train_sents = train_sents + batch_l
      local time_taken = timer:time().real - start_time
      if i % opt.print_every == 0 then
	local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
				    epoch, i, data:size(), batch_l, opt.learning_rate)
	stats = stats .. string.format('NLL: %.4f, Acc: %.4f, ',
				       train_loss/train_sents, train_num_correct/train_sents)
	stats = stats .. string.format('Training: %d total tokens/sec',
				       (num_words_target+num_words_source) / time_taken)
	print(stats)
      end
    end
    return train_loss, train_sents, train_num_correct
  end
  local best_val_perf = 0
  local test_perf = 0
  for epoch = 1, opt.epochs do
    local total_loss, total_sents, total_correct = train_batch(train_data, epoch)
    local train_score = total_correct/total_sents
    print('Train', train_score)
    opt.train_perf[#opt.train_perf + 1] = train_score
    local score = eval(valid_data)
    local savefile = string.format('%s.t7', opt.savefile)            
    if score > best_val_perf then
      best_val_perf = score
      test_perf = eval(test_data)
      print('saving checkpoint to ' .. savefile)
      torch.save(savefile, {layers, opt})	 	 
    end
    opt.val_perf[#opt.val_perf + 1] = score
    print(opt.train_perf)
    print(opt.val_perf)
  end
  print("Best Val", best_val_perf)
  print("Test", test_perf)   
  -- save final model
  local savefile = string.format('%s_final.t7', opt.savefile)
  print('saving final model to ' .. savefile)
  for i = 1, #layers do
    layers[i]:double()
  end   
  torch.save(savefile, {layers, opt})
end

function eval(data)
  sent_encoder:evaluate()
  local nll = 0
  local num_sents = 0
  local num_correct = 0
  for i = 1, data:size() do
    local d = data[i]
    local target, source, batch_l, target_l, source_l, label = table.unpack(d)
    local word_vecs1 = word_vecs_enc1:forward(source) 	 
    local word_vecs2 = word_vecs_enc2:forward(target)
    set_size_encoder(batch_l, source_l, target_l,
		     opt.word_vec_size, opt.hidden_size, all_layers)
    local  pred_input = {word_vecs1, word_vecs2}
    local pred_label = sent_encoder:forward(pred_input)
    local loss = disc_criterion:forward(pred_label, label)
    local _, pred_argmax = pred_label:max(2)
    num_correct = num_correct + pred_argmax:double():view(batch_l):eq(label:double()):sum()
    num_sents = num_sents + batch_l
    nll = nll + loss
  end
  local acc = num_correct/num_sents
  print("Acc", acc)
  print("NLL", nll / num_sents)
  collectgarbage()
  return acc
end

function main() 
  -- parse input params
  opt = cmd:parse(arg)
  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)      
  end
  
  -- Create the data loader class.
  print('loading data...')

  train_data = data.new(opt, opt.data_file)   
  valid_data = data.new(opt, opt.val_data_file)
  test_data = data.new(opt, opt.test_data_file)
  print('done!')
  print(string.format('Source vocab size: %d, Target vocab size: %d',
		      train_data.source_size, train_data.target_size))   
  opt.max_sent_l_src = train_data.source:size(2)
  opt.max_sent_l_targ = train_data.target:size(2)
  if opt.max_batch_l == '' then
    opt.max_batch_l = train_data.batch_l:max()
  end
  
  print(string.format('Source max sent len: %d, Target max sent len: %d',
		      train_data.source:size(2), train_data.target:size(2)))   
  
  -- Build model
  word_vecs_enc1 = nn.LookupTable(train_data.source_size, opt.word_vec_size)
  word_vecs_enc2 = nn.LookupTable(train_data.target_size, opt.word_vec_size)
  sent_encoder = make_sent_encoder(opt.word_vec_size, opt.hidden_size,
				   train_data.label_size, opt.dropout)	 

  disc_criterion = nn.ClassNLLCriterion()
  disc_criterion.sizeAverage = false
  layers = {word_vecs_enc1, word_vecs_enc2, sent_encoder}

  layer_etas = {}
  optStates = {}   
  for i = 1, #layers do
    layer_etas[i] = opt.learning_rate -- can have layer-specific lr, if desired
    optStates[i] = {}
  end
  
  if opt.gpuid >= 0 then
    for i = 1, #layers do	 
      layers[i]:cuda()
    end
    disc_criterion:cuda()
  end

  -- these layers will be manipulated during training
  all_layers = {}   
  sent_encoder:apply(get_layer)
  train(train_data, valid_data)
end

main()
