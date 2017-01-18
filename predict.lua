require 'nn'
require 'string'
require 'hdf5'
require 'nngraph'
require 'models.lua'

stringx = require('pl.stringx')

cmd = torch.CmdLine()

-- file location
cmd:option('-model', '', [[Path to model .t7 file]])
cmd:option('-sent1_file', '',[[Source sequence to decode (one line per sequence)]])
cmd:option('-sent2_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-word_dict', '', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-label_dict', '', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-gpuid',  -1, [[ID of the GPU to use (-1 = use CPU)]])
opt = cmd:parse(arg)

function idx2key(file)   
   local f = io.open(file,'r')
   local t = {}
   for line in f:lines() do
      local c = {}
      for w in line:gmatch'([^%s]+)' do
	 table.insert(c, w)
      end
      t[tonumber(c[2])] = c[1]
   end   
   return t
end

function flip_table(u)
   local t = {}
   for key, value in pairs(u) do
      t[value] = key
   end
   return t   
end

function sent2wordidx(sent, word2idx, start_symbol)
   local t = {}
   local u = {}
   table.insert(t, START)
   for word in sent:gmatch'([^%s]+)' do
      local idx = word2idx[word] or UNK 
      table.insert(t, idx)
   end
   return torch.LongTensor(t)
end

function wordidx2sent(sent, idx2word)
   local t = {}
   for i = 1, sent:size(1) do -- skip START and END
     table.insert(t, idx2word[sent[i]])	 
   end
   return table.concat(t, ' ')
end

function main()
   -- some globals
   PAD = 1; UNK = 2; START = 3; END = 4
   PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
   assert(path.exists(opt.model), 'model does not exist')
   
   -- parse input params
   opt = cmd:parse(arg)
   if opt.gpuid >= 0 then
      require 'cutorch'
      require 'cunn'
   end      
   print('loading ' .. opt.model .. '...')
   checkpoint = torch.load(opt.model)
   print('done!')
   model, model_opt = table.unpack(checkpoint)  
   -- load model and word2idx/idx2word dictionaries
   for i = 1, #model do
      model[i]:evaluate()
   end
   word_vecs_enc1 = model[1]
   word_vecs_enc2 = model[2]
   sent_encoder = model[3]
   all_layers = {}
   sent_encoder:apply(get_layer)
   idx2word = idx2key(opt.word_dict)
   word2idx = flip_table(idx2word)
   idx2label = idx2key(opt.label_dict)
   if opt.gpuid >= 0 then
      cutorch.setDevice(opt.gpuid)
      for i = 1, #model do
	model[i]:double():cuda()
      end
   end
   local sent1_file = io.open(opt.sent1_file, 'r')
   local sent2_file = io.open(opt.sent2_file, 'r')
   local out_file = io.open(opt.output_file,'w')
   local sent1 = {}
   local sent2 = {}
   for line in sent1_file:lines() do
     table.insert(sent1, sent2wordidx(line, word2idx))
   end
   for line in sent2_file:lines() do
     table.insert(sent2, sent2wordidx(line, word2idx))
   end
   assert(#sent1 == #sent2, 'number of sentences in sent1_file and sent2_file do not match')
   for i = 1, # sent1 do
     print('----SENTENCE PAIR ' .. i .. '----')
     print('SENT 1: ' .. wordidx2sent(sent1[i], idx2word))
     print('SENT 2: ' .. wordidx2sent(sent2[i], idx2word))
     local sent1_l = sent1[i]:size(1)
     local sent2_l = sent2[i]:size(1)
     local word_vecs1 = word_vecs_enc1:forward(sent1[i]:view(1, sent1_l))
     local word_vecs2 = word_vecs_enc2:forward(sent2[i]:view(1, sent2_l))
     set_size_encoder(1, sent1_l, sent2_l, model_opt.word_vec_size,
		      model_opt.hidden_size, all_layers)
     local pred = sent_encoder:forward({word_vecs1, word_vecs2})
     local _, pred_argmax = pred:max(2)
     local label_str = idx2label[pred_argmax[1][1]]
     print('PRED: ' .. label_str)
     out_file:write(label_str ..  '\n')     
   end
   out_file:close()
end
main()

