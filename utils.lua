function adagrad_step(x, dfdx, lr, state)
  if not state.var then
    state.var  = torch.Tensor():typeAs(x):resizeAs(x):zero():add(0.1)
    --adding 0.1 above is to be consistent with tensorflow
    state.std = torch.Tensor():typeAs(x):resizeAs(x)
  end
  state.var:addcmul(1, dfdx, dfdx)
  state.std:sqrt(state.var)
  x:addcdiv(-lr, dfdx, state.std)
end

function adam_step(x, dfdx, lr, state)
  local beta1 = state.beta1 or 0.9
  local beta2 = state.beta2 or 0.999
  local eps = state.eps or 1e-8

  state.t = state.t or 0
  state.m = state.m or x.new(dfdx:size()):zero()
  state.v = state.v or x.new(dfdx:size()):zero()
  state.denom = state.denom or x.new(dfdx:size()):zero()

  state.t = state.t + 1
  state.m:mul(beta1):add(1-beta1, dfdx)
  state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
  state.denom:copy(state.v):sqrt():add(eps)

  local bias1 = 1-beta1^state.t
  local bias2 = 1-beta2^state.t
  local stepSize = lr * math.sqrt(bias2)/bias1
  x:addcdiv(-stepSize, state.m, state.denom)
  
end

function adadelta_step(x, dfdx, lr, state)
  local rho = state.rho or 0.9
  local eps = state.eps or 1e-6
  state.var = state.var or x.new(dfdx:size()):zero()
  state.std = state.std or x.new(dfdx:size()):zero()
  state.delta = state.delta or x.new(dfdx:size()):zero()
  state.accDelta = state.accDelta or x.new(dfdx:size()):zero()
  state.var:mul(rho):addcmul(1-rho, dfdx, dfdx)
  state.std:copy(state.var):add(eps):sqrt()
  state.delta:copy(state.accDelta):add(eps):sqrt():cdiv(state.std):cmul(dfdx)
  x:add(-lr, state.delta)
  state.accDelta:mul(rho):addcmul(1-rho, state.delta, state.delta)   
end
