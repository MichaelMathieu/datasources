require 'datasources.datasource'
require 'paths'

local SVHNDatasource, parent = torch.class('SVHNDatasource', 'ClassDatasource')

function SVHNDatasource:__init()
   parent.__init(self)
   local raw_sets = {
      train = torch.load('/misc/vlgscratch3/LecunGroup/michael/datasets/svhn/train_32x32.t7b'),
      test = torch.load('/misc/vlgscratch3/LecunGroup/michael/datasets/svhn/test_32x32.t7b'),
      extra = torch.load('/misc/vlgscratch3/LecunGroup/michael/datasets/svhn/extra_32x32.t7b')
   }

   self.data, self.labels = {}, {}
   for k, v in pairs(raw_sets) do
      self.data[k] = v.X:type(torch.getdefaulttensortype())
      self.labels[k] = v.y:squeeze(1)
   end
   self:center(self.data.train, self.data)

   self.nChannels, self.nClasses = self.data.train:size(2), 10
   self.h, self.w = self.data.train:size(3), self.data.train:size(4)
end

function SVHNDatasource:nextBatch(batchSize, set)
   self.output_cpu:resize(batchSize, self.nChannels, self.h, self.w)
   self.labels_cpu:resize(batchSize)
   if set == 'train+extra' then
      local p = self.data.extra:size(1)/(self.data.train:size(1)+self.data.extra:size(1))
      for i = 1, batchSize do
	 local rndset = torch.bernoulli(p)
	 local this_set = (rndset == 1) and 'extra' or 'train'
	 local idx = torch.random(self.data[this_set]:size(1))
	 self.output_cpu[i]:copy(self.data[this_set][idx])
	 self.labels_cpu[i] = self.labels[this_set][idx]
      end
   else
      for i = 1, batchSize do
	 local idx = torch.random(self.data[set]:size(1))
	 self.output_cpu[i]:copy(self.data[set][idx])
	 self.labels_cpu[i] = self.labels[set][idx]
      end
   end
   return self:typeResults(self.output_cpu, self.labels_cpu)
end

function SVHNDatasource:nextIteratedBatchPerm(batchSize, set, idx, perm)
   error("todo")
end

function SVHNDatasource:nextIteratedBatch(batchSize, set, idx)
   assert(idx > 0)
   if idx*batchSize > self.data[set]:size(1) then
      return nil
   else
      return self:typeResults(self.data[set]:narrow(1, (idx-1)*batchSize+1, batchSize),
			      self.labels[set]:narrow(1, (idx-1)*batchSize+1, batchSize))
   end
end
