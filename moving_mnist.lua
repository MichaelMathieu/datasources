--[[ params:
   nDigits
   nFrames
   sigma
   w, h
   --TODO: this is not really a class datasource
--]]

require 'datasources.mnist'

local MovingMNISTDatasource, parent = torch.class('MovingMNISTDatasource', 'ClassDatasource')

function MovingMNISTDatasource:__init(params)
   parent.__init(self)
   self.mnist = MNISTDatasource()
   self.nDigits = params.nDigits or 2
   self.nFrames = params.nFrames or 3
   self.sigma = params.sigma or 1
   self.h = params.h or 64
   self.w = params.w or 64
   self.digith, self.digitw = self.mnist.h, self.mnist.w
   assert((self.digith <= self.h) and (self.digitw <= self.w))
   self.nChannels = self.mnist.nChannels
end

local function movebounce(x, dx, w)
   if x < 1 then x = 1 end
   if x > w then x = w end
   if x + dx < 1 then
      x = 2 - x - dx
      dx = -dx
   elseif x + dx > w then
      x = 2*w - x - dx
      dx = -dx
   else
      x = x + dx
   end
   if x < 1 then x = 1 end
   if x > w then x = w end
   return x, dx
end

function MovingMNISTDatasource:nextBatch(batchSize, set)
   self.output_cpu:resize(batchSize, self.nFrames, self.nChannels, self.h, self.w)
   self.labels_cpu:resize(batchSize, self.nDigits)
   self.output_cpu:zero()
   for i = 1, self.nDigits do
      local batch, labels = self.mnist:nextBatch(batchSize, set)
      batch:add(1) -- put it in [0, 2]
      self.labels_cpu:select(2, i):copy(labels)
      for j = 1, batchSize do
	 local y, x = torch.random(self.h - self.digith + 1), torch.random(self.w - self.digitw + 1)
	 local dy, dx = torch.normal()*self.sigma, torch.normal()*self.sigma
	 for k = 1, self.nFrames do
	    self.output_cpu[j][k]:narrow(2, y, self.digith):narrow(3, x, self.digitw):add(batch[j])
	    y, dy = movebounce(y, dy, self.h - self.digith + 1)
	    x, dx = movebounce(x, dx, self.w - self.digitw + 1)
	 end
      end
   end
   self.output_cpu:clamp(0, 2):add(-1)
   return self:typeResults(self.output_cpu, self.labels_cpu)
end