require 'datasources.datasource'
require 'paths'
require 'image'
require 'math'

local function round(x)
   return math.floor(x+0.5)
end

local TransformDatasource, parent = torch.class('TransformDatasource', 'ClassDatasource')

function TransformDatasource:__init(datasource, params)
   parent.__init(self)
   self.datasource = datasource
   self.nChannels, self.nClasses = datasource.nChannels, datasource.nClasses
   if params.crop then
      assert(#(params.crop) == 2)
      self.h, self.w = params.crop[1], params.crop[2]
   else
      self.h, self.w = datasource.h, datasource.w
   end

   if self.datasource.tensortype == 'torch.CudaTensor' then
      print("Warning: TransformDatasource used with a cuda datasource. Might break")
   end

   self.params = {
      crop = params.crop or {self.h, self.w},
   }
end

local function flatten3d(x)
   -- if x is a video, flatten it
   if x:dim() == 4 then
      return x:view(x:size(1)*x:size(2), x:size(3), x:size(4))
   else
      assert(x:dim() == 3)
      return x
   end
end

local function dimxy(x)
   assert((x:dim() == 3) or (x:dim() == 4))
   if x:dim() == 4 then
      return 3, 4
   else
      return 2, 3
   end
end

local function crop(patch, hTarget, wTarget, minMotion, minMotionNTries)
   local dimy, dimx = dimxy(patch)
   local h, w = patch:size(dimy), patch:size(dimx)
   assert((h >= hTarget) and (w >= wTarget))
   if (h == hTarget) and (w == wTarget) then
      return patch
   else
      if minMotion then
	 assert(patch:dim() == 4)
	 local x, y
	 for i = 1, minMotionNTries do
	    y = torch.random(1, h-hTarget+1)
	    x = torch.random(1, w-wTarget+1)
	    local cropped = patch:narrow(dimy, y, hTarget):narrow(dimx, x, wTarget)
	    if (cropped[-1] - cropped[-2]):norm() > math.sqrt(minMotion * cropped[-1]:nElement()) then
	       break
	    end
	 end
	 return patch:narrow(dimy, y, hTarget):narrow(dimx, x, wTarget)
      else
	 local y = torch.random(1, h-hTarget+1)
	 local x = torch.random(1, w-wTarget+1)
	 return patch:narrow(dimy, y, hTarget):narrow(dimx, x, wTarget)
      end
   end
end

local input2_out = torch.Tensor()
function TransformDatasource:transform(input, target)
   if input:dim() == 4 then
      input2_out:resize(input:size(1), input:size(2),
			self.params.crop[1], self.params.crop[2])
   else
      input2_out:resize(input:size(1), input:size(2), input:size(3),
			self.params.crop[1], self.params.crop[2])      
   end
   for i = 1, input:size(1) do
      local x = input[i]
      x = crop(x, self.params.crop[1], self.params.crop[2])
      input2_out[i]:copy(x)
   end
   return input2_out, target
end

function TransformDatasource:nextBatch(batchSize, set)
   local input, target = self.datasource:nextBatch(batchSize, set)
   local input2, target2 = self:transform(input, target)
   return self:typeResults(input2, target2)
end

function TransformDatasource:orderedIterator(batchSize, set, extraargs)
   local it = self.datasource:orderedIterator(batchSize, set, extraargs)
   return function()
      local input, target = it()
      if input ~= nil then
	 local input2, target2 = self:transform(input, target)
	 return self:typeResults(input2, target2)
      else
	 return nil
      end
   end
end
