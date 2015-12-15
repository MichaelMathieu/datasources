require 'datasource'
require 'paths'

local SVHNDatasource, parent = torch.class('SVHNDatasource', 'Datasource')

function SVHNDatasource:__init()
   parent.__init(self)
   self.tensortype = torch.getdefaulttensortype()
   self.train_set = torch.load('/misc/vlgscratch3/LecunGroup/michael/datasets/svhn/train_32x32.t7b')
   self.test_set = torch.load('/misc/vlgscratch3/LecunGroup/michael/datasets/svhn/test_32x32.t7b')
   self.extra_set = torch.load('/misc/vlgscratch3/LecunGroup/michael/datasets/svhn/extra_32x32.t7b')
   self.train_set.data = self.train_set.X
   self.train_set.labels = self.train_set.y:squeeze(1)
   self.test_set.data = self.test_set.X
   self.test_set.labels = self.test_set.y:squeeze(1)
   self.extra_set.data = self.extra_set.X
   self.extra_set.labels = self.extra_set.y:squeeze(1)
   self.train_set.X, self.train_set.y = nil, nil
   self.test_set.X, self.test_set.y = nil, nil
   self.extra_set.X, self.extra_set.y = nil, nil
   self.nChannels = 3
   self.h = 32
   self.w = 32
   self.nClasses = 10
   self.output = torch.Tensor()
   self.labels = torch.LongTensor()
   self.nSamples = {['train'] = self.train_set.data:size(1),
		    ['test'] = self.test_set.data:size(1),
		    ['extra'] = self.extra_set.data:size(1)}

   self.train_set.data = self.train_set.data:float()
   self.test_set.data = self.test_set.data:float()
   self.extra_set.data = self.extra_set.data:float()

   self.mean = torch.Tensor(3)
   for i = 1, 3 do
      self.mean[i] = self.train_set.data[{{},i}]:mean()
      self.train_set.data[{{},i}]:add(-self.mean[i])
      self.test_set.data[{{},i}]:add(-self.mean[i])
      self.extra_set.data[{{},i}]:add(-self.mean[i])
   end
   self.std = torch.Tensor(3)
   for i = 1, 3 do
      self.std[i] = self.train_set.data[{{},i}]:std()
      self.train_set.data[{{},i}]:div(self.std[i])
      self.test_set.data[{{},i}]:div(self.std[i])
      self.extra_set.data[{{},i}]:div(self.std[i])
   end
end

function SVHNDatasource:produceResults(output, labels)
   if self.tensortype == 'torch.CudaTensor' then
      self.output_gpu:resize(output:size()):copy(output)
      self.labels_gpu:resize(labels:size()):copy(labels)
      return {self.output_gpu, self.labels_gpu}
   else
      return {output, labels}
   end
end

function SVHNDatasource:nextBatch(batchSize, set)
   if set == 'train+extra' then
      self.output:resize(batchSize, self.nChannels, self.h, self.w)
      self.labels:resize(batchSize)
      local p = self.extra_set.data:size(1)/(self.train_set.data:size(1)+self.extra_set.data:size(1))
      local this_set
      for i = 1, batchSize do
	 local set = torch.bernoulli(p)
	 if set == 1 then
	    this_set = self.extra_set
	 else
	    this_set = self.train_set
	 end
	 local idx = torch.random(this_set.data:size(1))
	 self.output[i]:copy(this_set.data[idx])
	 --TODO: more GPU friendly
	 self.labels[i] = this_set.labels[idx]
      end
      return self:produceResults(self.output, self.labels)
   else
      local this_set
      if set == 'train' then
	 this_set = self.train_set
      elseif set == 'test' then
	 this_set = self.test_set
      elseif set == 'extra' then
	 this_set = self.extra_set
      else
	 error('set must be [train|test|extra|train+extra] for SVHN')
      end
      self.output:resize(batchSize, self.nChannels, self.h, self.w)
      self.labels:resize(batchSize)
      for i = 1, batchSize do
	 local idx = torch.random(this_set.data:size(1))
	 self.output[i]:copy(this_set.data[idx])
	 --TODO: more GPU friendly
	 self.labels[i] = this_set.labels[idx]
      end
      return self:produceResults(self.output, self.labels)
   end
end

function SVHNDatasource:nextIteratedBatchPerm(batchSize, set, idx, perm)
   local this_set
   if set == 'train' then
      this_set = self.train_set
   elseif set == 'test' then
      this_set = self.test_set
   elseif set == 'extra' then
      this_set = self.extra_set
   else
      error('set must be [train|test|extra] for SVHN')
   end
   self.output:resize(batchSize, self.nChannels, self.h, self.w)
   self.labels:resize(batchSize)
   for i = 1, batchSize do
      local idx1 = (idx-1)*batchSize+i
      if idx1 > perm:size(1) then
	 return nil
      end
      local idx2 = perm[idx1]
      self.output[i]:copy(this_set.data[idx2])
      --TODO: more GPU friendly
      self.labels[i] = this_set.labels[idx2]
   end
   return self:produceResults(self.output, self.labels)
end

function SVHNDatasource:nextIteratedBatch(batchSize, set, idx)
   assert(idx > 0)
   local this_set
   if set == 'train' then
      this_set = self.train_set
   elseif set == 'test' then
      this_set = self.test_set
   elseif set == 'extra' then
      this_set = self.extra_set
   else
      error('set must be [train|test|extra] for SVHN')
   end
   if idx*batchSize > this_set.data:size(1) then
      return nil
   else
      return self:produceResults(this_set.data:narrow(1, (idx-1)*batchSize+1, batchSize),
				 this_set.labels:narrow(1, (idx-1)*batchSize+1, batchSize))
   end
end

function SVHNDatasource:type(typ)
   self.tensortype = typ
   if typ == 'torch.CudaTensor' then
      self.output_gpu = torch.CudaTensor()
      self.labels_gpu = torch.CudaTensor()
   else
      self.output_gpu = nil
      self.labels_gpu = nil
      collectgarbage()
   end
end