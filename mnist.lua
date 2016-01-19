--[[
   classes are from 1 to 10 where 1 corresponds to the digit 0
   and 10 to the digit 9 (UNLIKE SVHN!!)
--]]

require 'datasources.datasource'
require 'paths'

local MNISTDatasource, parent = torch.class('MNISTDatasource', 'ClassDatasource')

function MNISTDatasource:__init()
   parent.__init(self)
   local raw_sets = {
      train = torch.load('/misc/vlgscratch3/LecunGroup/michael/datasets/mnist/train_28x28.th7'),
      test = torch.load('/misc/vlgscratch3/LecunGroup/michael/datasets/mnist/test_28x28.th7')
   }
   
   self.data = {train = raw_sets.train.data:type(torch.getdefaulttensortype()),
		test = raw_sets.test.data:type(torch.getdefaulttensortype())}
   self.labels = {train = raw_sets.train.labels, test = raw_sets.test.labels}
   self:normalize(self.data.train, self.data)

   self.nChannels, self.nClasses = self.data.train:size(2), 10
   self.h, self.w = self.data.train:size(3), self.data.train:size(4)
end

function MNISTDatasource:nextBatch(batchSize, set)
   assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
   assert(self.data[set] ~= nil, 'Unknown set ' .. set)
   self.output_cpu:resize(batchSize, self.nChannels, self.h, self.w)
   self.labels_cpu:resize(batchSize)
   for i = 1, batchSize do
      local idx = torch.random(self.data[set]:size(1))
      self.output_cpu[i]:copy(self.data[set][idx])
      self.labels_cpu[i] = self.labels[set][idx]
   end
   return self:typeResults(self.output_cpu, self.labels_cpu)
end

function MNISTDatasource:orderedIterator(batchSize, set)
   assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
   assert(self.data[set] ~= nil, 'Unknown set ' .. set)
   local idx = 1
   return function()
      if idx*batchSize > self.data[set]:size(1) then
	 return nil
      else
	 local outputs = self.data[set]:narrow(1, (idx-1)*batchSize+1, batchSize)
	 local labels = self.labels[set]:narrow(1, (idx-1)*batchSize+1, batchSize)
	 idx = idx + 1
	 return self:typeResults(outputs, labels)
      end
   end
end
