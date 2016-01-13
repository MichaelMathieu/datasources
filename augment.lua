require 'datasources.datasource'
require 'paths'
require 'image'
require 'math'

local function round(x)
   return math.floor(x+0.5)
end

local AugmentDatasource, parent = torch.class('AugmentDatasource', 'ClassDatasource')

function AugmentDatasource:__init(datasource, params)
   parent.__init(self)
   self.datasource = datasource
   self.nChannels, self.nClasses = datasource.nChannels, datasource.nClasses
   if params.cropSize then
      assert(#(params.cropSize) == 2)
      self.h, self.w = params.cropSize[1], params.cropSize[2]
   else
      self.h, self.w = datasource.h, datasource.w
   end

   if self.datasource.tensortype == 'torch.CudaTensor' then
      print("Warning: AugmentDatasource used with a cuda datasource. Might break")
   end

   self.params = {
      flip = params.flip or 0, --1 for vflip, 2 for hflip, 3 for both
      crop = params.crop or {self.h, self.w},
      scaleup = params.scaleup or 1,
      rotate = params.rotate or 0,
   }
end

local flip_out1, flip_out2 = torch.Tensor(), torch.Tensor()
local function flip(patch, mode)
   local out = patch
   if (mode == 1) or (mode == 3) then
      if torch.bernoulli(0.5) == 1 then
	 flip_out1:typeAs(out):resizeAs(out)
	 image.vflip(flip_out1, out)
	 out = flip_out1
      end
   end
   if (mode == 2) or (mode == 3) then
      if torch.bernoulli(0.5) == 1 then
	 flip_out2:typeAs(out):resizeAs(out)
	 image.hflip(flip_out2, out)
	 out = flip_out2
      end
   end
   return out
end

local function crop(patch, hTarget, wTarget)
   local h, w = patch:size(2), patch:size(3)
   assert((h >= hTarget) and (w >= wTarget))
   if (h == hTarget) and (w == wTarget) then
      return patch
   else
      local y = torch.random(1, h-hTarget+1)
      local x = torch.random(1, w-wTarget+1)
      return patch:narrow(2, y, hTarget):narrow(3, x, wTarget)
   end
end

local scaleup_out = torch.Tensor()
local function scaleup(patch, maxscale, mode)
   mode = mode or 'bilinear'
   assert(maxscale >= 1)
   local h, w = patch:size(2), patch:size(3)
   local maxH, maxW = round(h*maxscale), round(w*maxscale)
   if (maxH == h) and (maxW == w) then
      return patch
   else
      local scaleH = torch.random(h, maxH)
      local scaleW = torch.random(w, maxW)
      scaleup_out:typeAs(patch):resize(patch:size(1), scaleH, scaleW)
      return image.scale(scaleup_out, patch, mode)
   end
end

local rotate_out = torch.Tensor()
local function rotate(patch, thetamax, mode)
   mode = mode or 'bilinear'
   assert(thetamax >= 0)
   if thetamax == 0 then
      return patch
   else
      local theta = torch.uniform(-thetamax, thetamax)
      rotate_out:typeAs(patch):resizeAs(patch)
      return image.rotate(rotate_out, patch, theta, mode)
   end
end

function AugmentDatasource:nextBatch(batchSize, set)
   local data = self.datasource:nextBatch(batchSize, set)
   local input, target = data[1], data[2]
   for i = 1, batchSize do
      local x = input[i]
      x = flip(x, self.params.flip)
      x = rotate(x, self.params.rotate)
      x = scaleup(x, self.params.scaleup)
      x = crop(x, self.params.crop[1], self.params.crop[2])
      input[i]:copy(x)
   end
   return self:typeResults(input, target)
end

--This has NO data augmentation (you can't iterate over augmented data, it's infinite)
function AugmentDatasource:nextIteratedBatch(batchSize, set, idx)
   local data = self.datasource:nextIteratedBatch(batchSize, set, idx)
   if data == nil then
      return nil
   else
      return self:typeResults(data[1], data[2])
   end
end
