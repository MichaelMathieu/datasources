--[[ params:
   nFrames
   nBalls
   sigma
   w, h
   size,
   --TODO: this is not really a class datasource
--]]

require 'datasources.datasource'

local MovingBallsDatasource, parent = torch.class('MovingBallsDatasource', 'ClassDatasource')

function MovingBallsDatasource:__init(params)
   parent.__init(self)
   self.nFrames = params.nFrames or 3
   self.nBalls = params.nBalls or 2
   self.sigma = params.sigma or 1
   self.h = params.h or 64
   self.w = params.w or 64
   self.size = params.size or 16
   self.ball = torch.Tensor(self.size, self.size):zero()
   for x = 1, self.size do
      for y = 1, self.size do
	 local dx = x-(self.size-1)/2-1
	 local dy = y-(self.size-1)/2-1
	 if dx*dx + dy*dy < self.size*self.size/4 then
	    self.ball[y][x] = 1
	 end
      end
   end
   self.nChannels = 1
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

function MovingBallsDatasource:nextBatch(batchSize, set)
   self.output_cpu:resize(batchSize, self.nFrames, self.nChannels, self.h, self.w)
   self.output_cpu:zero()
   for i = 1, self.nBalls do
      for j = 1, batchSize do
	 local y, x = torch.random(self.h - self.size + 1), torch.random(self.w - self.size + 1)
	 local dy, dx = torch.normal()*self.sigma, torch.normal()*self.sigma
	 for k = 1, self.nFrames do
	    self.output_cpu[j][k]:narrow(2, y, self.size):narrow(3, x, self.size):add(self.ball)
	    y, dy = movebounce(y, dy, self.h - self.size + 1)
	    x, dx = movebounce(x, dx, self.w - self.size + 1)
	 end
      end
   end
   self.output_cpu:mul(2):clamp(0, 2):add(-1)
   return self:typeResults(self.output_cpu, self.labels_cpu)
end