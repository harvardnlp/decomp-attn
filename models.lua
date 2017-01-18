function make_sent_encoder(input_size, hidden_size, num_labels, dropout)
   local sent_l1 = 5 -- sent_l1, sent_l2, and batch_l are default values that will change 
   local sent_l2 = 10
   local batch_l = 1
   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local input1 = inputs[1] -- batch_l x sent_l1 x input_size
   local input2 = inputs[2] --batch_l x sent_l2 x input_size
   
   local input1_proj, input2_proj, size
   local proj1 = nn.Linear(input_size, hidden_size, false)
   local proj2 = nn.Linear(input_size, hidden_size, false)
   proj1.name = 'proj1'
   proj2.name = 'proj2'
   local input1_proj_view = nn.View(batch_l*sent_l1, input_size)
   local input2_proj_view = nn.View(batch_l*sent_l2, input_size)
   local input1_proj_unview = nn.View(batch_l, sent_l1, hidden_size)
   local input2_proj_unview = nn.View(batch_l, sent_l2, hidden_size)   
   input1_proj_view.name = 'input1_proj_view'
   input2_proj_view.name = 'input2_proj_view'
   input1_proj_unview.name = 'input1_proj_unview'
   input2_proj_unview.name = 'input2_proj_unview'
   input1_proj = input1_proj_unview(proj1(input1_proj_view(input1))) 
   input2_proj = input2_proj_unview(proj2(input2_proj_view(input2)))      
   size = hidden_size      

   local f1 = nn.Sequential()
   f1:add(nn.Dropout(dropout))      
   f1:add(nn.Linear(size, hidden_size))
   f1:add(nn.ReLU())
   f1:add(nn.Dropout(dropout))
   f1:add(nn.Linear(hidden_size, hidden_size))
   f1:add(nn.ReLU())
   f1.name = 'f1'
   local f2 = nn.Sequential()
   f2:add(nn.Dropout(dropout))   
   f2:add(nn.Linear(size, hidden_size))
   f2:add(nn.ReLU())
   f2:add(nn.Dropout(dropout))
   f2:add(nn.Linear(hidden_size, hidden_size))
   f2:add(nn.ReLU())
   f2.name = 'f2'
   local input1_view = nn.View(batch_l*sent_l1, size)
   local input2_view = nn.View(batch_l*sent_l2, size)
   local input1_unview = nn.View(batch_l, sent_l1, hidden_size)
   local input2_unview = nn.View(batch_l, sent_l2, hidden_size)   
   input1_view.name = 'input1_view'
   input2_view.name = 'input2_view'
   input1_unview.name = 'input1_unview'
   input2_unview.name = 'input2_unview'

   local input1_hidden = input1_unview(f1(input1_view(input1_proj)))
   local input2_hidden = input2_unview(f2(input2_view(input2_proj)))
   local scores1 = nn.MM()({input1_hidden,
			    nn.Transpose({2,3})(input2_hidden)}) -- batch_l x sent_l1 x sent_l2
   local scores2 = nn.Transpose({2,3})(scores1) -- batch_l x sent_l2 x sent_l1

   local scores1_view = nn.View(batch_l*sent_l1, sent_l2)
   local scores2_view = nn.View(batch_l*sent_l2, sent_l1)
   local scores1_unview = nn.View(batch_l, sent_l1, sent_l2)
   local scores2_unview = nn.View(batch_l, sent_l2, sent_l1)
   scores1_view.name = 'scores1_view'
   scores2_view.name = 'scores2_view'
   scores1_unview.name = 'scores1_unview'
   scores2_unview.name = 'scores2_unview'
  
   local prob1 = scores1_unview(nn.SoftMax()(scores1_view(scores1))) 
   local prob2 = scores2_unview(nn.SoftMax()(scores2_view(scores2)))
  
   local input2_soft = nn.MM()({prob1, input2_proj}) -- batch_l x sent_l1 x input_size
   local input1_soft = nn.MM()({prob2, input1_proj}) -- batch_l x sent_l2 x input_size

   local input1_combined = nn.JoinTable(3)({input1_proj ,input2_soft}) -- batch_l x sent_l1 x input_size*2
   local input2_combined = nn.JoinTable(3)({input2_proj,input1_soft}) -- batch_l x sent_l2 x input_size*2
   local new_size = size*2
   local input1_combined_view = nn.View(batch_l*sent_l1, new_size)
   local input2_combined_view = nn.View(batch_l*sent_l2, new_size)
   local input1_combined_unview = nn.View(batch_l, sent_l1, hidden_size)
   local input2_combined_unview = nn.View(batch_l, sent_l2, hidden_size)
   input1_combined_view.name = 'input1_combined_view'
   input2_combined_view.name = 'input2_combined_view'
   input1_combined_unview.name = 'input1_combined_unview'
   input2_combined_unview.name = 'input2_combined_unview'

   local g1 = nn.Sequential()
   g1:add(nn.Dropout(dropout))   
   g1:add(nn.Linear(new_size, hidden_size))
   g1:add(nn.ReLU())
   g1:add(nn.Dropout(dropout))      
   g1:add(nn.Linear(hidden_size, hidden_size))
   g1:add(nn.ReLU())
   g1.name = 'g1'
   local g2 = nn.Sequential()
   g2:add(nn.Dropout(dropout))
   g2:add(nn.Linear(new_size, hidden_size))
   g2:add(nn.ReLU())
   g2:add(nn.Dropout(dropout))         
   g2:add(nn.Linear(hidden_size, hidden_size))
   g2:add(nn.ReLU())
   g2.name = 'g2'
   local input1_output = input1_combined_unview(g1(input1_combined_view(input1_combined)))
   local input2_output = input2_combined_unview(g2(input2_combined_view(input2_combined)))
   input1_output = nn.Sum(2)(input1_output) -- batch_l x hidden_size
   input2_output = nn.Sum(2)(input2_output) -- batch_l x hidden_size     
   new_size = hidden_size*2
   
   local join_layer = nn.JoinTable(2)
   local input12_combined = join_layer({input1_output, input2_output})
   join_layer.name = 'join'
   local out_layer = nn.Sequential()
   out_layer:add(nn.Dropout(dropout))
   out_layer:add(nn.Linear(new_size, hidden_size))
   out_layer:add(nn.ReLU())
   out_layer:add(nn.Dropout(dropout))
   out_layer:add(nn.Linear(hidden_size, hidden_size))
   out_layer:add(nn.ReLU())
   out_layer:add(nn.Linear(hidden_size, num_labels))
   out_layer:add(nn.LogSoftMax())
   out_layer.name = 'out_layer'
   local out = out_layer(input12_combined)
   return nn.gModule(inputs, {out})
end

function get_layer(layer)
  if layer.name ~= nil then
    all_layers[layer.name] = layer
  end
end


function set_size_encoder(batch_l, sent_l1, sent_l2, input_size, hidden_size, t)
   local size = hidden_size
   t.input1_proj_view.size[1] = batch_l*sent_l1
   t.input1_proj_view.numElements = batch_l*sent_l1*input_size
   t.input2_proj_view.size[1] = batch_l*sent_l2
   t.input2_proj_view.numElements = batch_l*sent_l2*input_size

   t.input1_proj_unview.size[1] = batch_l
   t.input1_proj_unview.size[2] = sent_l1
   t.input1_proj_unview.numElements = batch_l*sent_l1*hidden_size
   t.input2_proj_unview.size[1] = batch_l
   t.input2_proj_unview.size[2] = sent_l2
   t.input2_proj_unview.numElements = batch_l*sent_l2*hidden_size

   t.input1_view.size[1] = batch_l*sent_l1
   t.input1_view.numElements = batch_l*sent_l1*size   
   t.input1_unview.size[1] = batch_l
   t.input1_unview.size[2] = sent_l1
   t.input1_unview.numElements = batch_l*sent_l1*hidden_size
   
   t.input2_view.size[1] = batch_l*sent_l2
   t.input2_view.numElements = batch_l*sent_l2*size
   t.input2_unview.size[1] = batch_l
   t.input2_unview.size[2] = sent_l2
   t.input2_unview.numElements = batch_l*sent_l2*hidden_size     
   
   t.scores1_view.size[1] = batch_l*sent_l1
   t.scores1_view.size[2] = sent_l2
   t.scores1_view.numElements = batch_l*sent_l1*sent_l2
   t.scores2_view.size[1] = batch_l*sent_l2
   t.scores2_view.size[2] = sent_l1
   t.scores2_view.numElements = batch_l*sent_l1*sent_l2

   t.scores1_unview.size[1] = batch_l
   t.scores1_unview.size[2] = sent_l1
   t.scores1_unview.size[3] = sent_l2
   t.scores1_unview.numElements = batch_l*sent_l1*sent_l2
   t.scores2_unview.size[1] = batch_l
   t.scores2_unview.size[2] = sent_l2 
   t.scores2_unview.size[3] = sent_l1  
   t.scores2_unview.numElements = batch_l*sent_l1*sent_l2

   t.input1_combined_view.size[1] = batch_l*sent_l1
   t.input1_combined_view.numElements = batch_l*sent_l1*2*size
   t.input2_combined_view.size[1] = batch_l*sent_l2
   t.input2_combined_view.numElements = batch_l*sent_l2*2*size     

   t.input1_combined_unview.size[1] = batch_l
   t.input1_combined_unview.size[2] = sent_l1
   t.input1_combined_unview.numElements = batch_l*sent_l1*hidden_size
   t.input2_combined_unview.size[1] = batch_l
   t.input2_combined_unview.size[2] = sent_l2
   t.input2_combined_unview.numElements = batch_l*sent_l2*hidden_size
end


