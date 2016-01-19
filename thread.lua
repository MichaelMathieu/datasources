--[[
   Note that it costs time to switch from set (train/test/valid)
   and change the batch size. If you intend to do it a lot, create
   multiple instances of datasources, with constant set/batchSize
   params:
   nDonkeys [4]
--]]

require 'datasources.datasource'
local threads = require 'threads'

local ThreadedDatasource, parent = torch.class('ThreadedDataset', 'ClassDatasource')

function ThreadedDatasource:__init(getDatasourceFun, params)
   parent.__init(self)
   self.nDonkeys = params.nDonkeys or 4
   threads.Threads.serialization('threads.sharedserialize')
   self.donkeys = threads.Threads(self.nDonkeys,
      function(threadid)
	 require 'torch'
	 threadid_t = threadid
	 datasource_t = getDatasourceFun()
      end)
   self.started = false
end

function ThreadedDatasource:type(typ)
   parent.type(self, typ)
   if typ == 'torch.CudaTensor' then
      self.output, self.labels = self.output_gpu, self.labels_gpu
   else
      self.output, self.labels = self.output_cpu, self.labels_cpu
   end
end   

end

function ThreadedDatasource:nextBatch(batchSize, set)
   local function addjob()
      self.donkeys:addjob(
	 function()
	    return datasource_t:nextBatch(batchSize, set)
	 end,
	 function(outputs, labels)
	    self.outputs:resizeAs(outputs):copy(outputs)
	    self.labels:resizeAs(labels):copy(labels)
	    self.last_config = {batchSize, set}
	 end)
   end
   if not self.started then
      self.donkeys:specific(false)
      for i = 1, self.nDonkeys do
	 addjob()
      end
   end
   self.last_config = {}
   while (self.last_config[1] ~= batchSize) or (self.last_config[2] ~= set) do
      self.donkeys:dojob()
      addjob()
   end
   return self.output, self.labels
end

function ThreadedDatasource:orderedIterator(batchSize, set)
   -- this one doesn't parallelize on more than one thread
   -- (this might be a TODO but seems hard)
   self.donkeys:specific(true)
   self.donkeys:synchronize()
   self.started = false
   self.donkeys:addjob(
      1, function() it_t = datasource_t:orderedIterator(batchSize, set) end)
   local finished = false
   local function addjob()
      self.donkeys:addjob(
	 1,
	 function()
	    return it_t()
	 end,
	 function(output, labels)
	    if output == nil then
	       finished = true
	    else
	       self.outputs:resizeAs(outputs):copy(outputs)
	       self.labels:resizeAs(labels):copy(labels)
	    end
	 end)
   end
   return function()
      self.donkeys:synchronize()
      if finished then
	 self.donkeys:addjob(1, function() it_t = nil collectgarbage() end)
	 self.donkeys:synchronize()
      else
	 addjob()
	 return self.outputs, self.labels
      end
   end
end