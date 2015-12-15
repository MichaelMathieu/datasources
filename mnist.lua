require 'datasource'
require 'paths'

local MNISTDatasource, parent = torch.class('MNISTDatasource', 'Datasource')

function MNISTDatasource:__init()
   parent.__init(self)
   if paths.dirp('/scratch/michael/datasets/mnist') then
      self.train_set = torch.load('/scratch/michael/datasets/mnist/train_28x28.th7')
      self.test_set = torch.load('/scratch/michael/datasets/mnist/test_28x28.th7')
   else
      self.train_set = torch.load('/misc/vlgscratch3/LecunGroup/michael/datasets/mnist/train_28x28.th7')
      self.test_set = torch.load('/misc/vlgscratch3/LecunGroup/michael/datasets/mnist/test_28x28.th7')
   end
   self.nChannels = 1
   self.h = 28
   self.w = 28
   self.output = torch.Tensor()
   self.labels = torch.LongTensor()

   self.train_set.data = self.train_set.data:float()
   self.test_set.data = self.test_set.data:float()

   self.mean = self.train_set.data:mean()
   self.train_set.data:add(-self.mean)
   self.test_set.data:add(-self.mean)
   self.std = self.train_set.data:std()
   self.train_set.data:div(self.std)
   self.test_set.data:div(self.std)
   
end

function MNISTDatasource:nextBatch(batchSize, set)
   local this_set
   if set == 'train' then
      this_set = self.train_set
   elseif set == 'test' then
      this_set = self.test_set
   else
      error('set must be [train|test] for MNIST')
   end
   self.output:resize(batchSize, self.nChannels, self.h, self.w)
   self.labels:resize(batchSize)
   for i = 1, batchSize do
      local idx = torch.random(this_set.data:size(1))
      self.output[i]:copy(this_set.data[idx])
      --TODO: more GPU friendly
      self.labels[i] = this_set.labels[idx]
   end
   return {self.output, self.labels}
end

function MNISTDatasource:nextIteratedBatch(batchSize, set, idx)
   local this_set
   if set == 'train' then
      this_set = self.train_set
   elseif set == 'test' then
      this_set = self.test_set
   else
      error('set must be [train|test] for CIFAR')
   end
   if idx*batchSize > this_set.data:size(1) then
      return nil
   else
      return {this_set.data:narrow(1, (idx-1)*batchSize+1, batchSize),
	      this_set.labels:narrow(1, (idx-1)*batchSize+1, batchSize)}
   end
end

function MNISTDatasource:type(typ)
   self.train_set.data = self.train_set.data:type(typ)
   self.test_set.data = self.test_set.data:type(typ)
   self.output = self.output:type(typ)
   if typ == 'torch.CudaTensor' then
      self.train_set.labels = self.train_set.labels:type(typ)
      self.test_set.labels = self.test_set.labels:type(typ)
      self.labels = self.labels:type(typ)
   else
      self.train_set.labels = self.train_set.labels:type('torch.LongTensor')
      self.test_set.labels = self.test_set.labels:type('torch.LongTensor')
      self.labels = self.labels:type('torch.LongTensor')
   end
end