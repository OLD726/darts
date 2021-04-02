import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

"""
************alpha的更新

step --> _backward_step_unrolled --> (_compute_unrolled_model ||  _hessian_vector_product) 
"""



def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    # 优化器，arch_parameters
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    # 计算ω，Ltrain
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      # 增加动量
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)

    # 计算
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta

    # 构建新网络模型
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    # 更新w后的unrolled_modle
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    # 计算Lval（w'）
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)
    unrolled_loss.backward()

    # 公式（7）
    # 计算dαLval（w',α）
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    # 计算dw‘Lval（w’，α）
    # 用于计算w+,w-
    vector = [v.grad.data for v in unrolled_model.parameters()]

    # 公式（8）
    # 计算(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    # 计算公式（7）
    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    # 更新α
    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)

    # 返回 参数更新为一次反向传播的值 的模型
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()

    # 计算w+
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    # 计算dαLtrain(w+,α)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    # 计算w-
    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    # 计算dαLtrain(w-,α)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

