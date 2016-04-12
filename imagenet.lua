--[[
   ImageNet 128x128 random crops (one dim is full)
   params:
     rootpath: root path. Must contain
     - a folder index (containing train.t7b and val.t7b)
     - a folder 128x128quality9075/ILSVRC2012_img_train
     - a folder 128x128quality9075/ILSVRC2012_img_val
--]]

require 'datasources.datasource'
require 'paths'
require 'image'
require 'math'

local ImageNetDatasource, parent = torch.class('ImageNetDatasource', 'ClassDatasource')

function ImageNetDatasource:__init(params)
   parent.__init(self)
   params = params or {}
   self.path = params.rootpath or '/misc/vlgscratch2/LecunGroup/xz558/public/imagenet/data/class/pp'
   self.dirs = {
      train = '128x128quality9075/ILSVRC2012_img_train',
      val = '128x128quality9075/ILSVRC2012_img_val'}
   self.inds = {
      train = torch.load(paths.concat(self.path, 'index/train.t7b')).files,
      val = torch.load(paths.concat(self.path, 'index/val.t7b')).files}
   
   self.nChannels, self.nClasses = 3, 1000
   self.h, self.w = 128, 128
end

function ImageNetDatasource:randomCrop(im)
   local h, w = im:size(2), im:size(3)
   if h > self.h then
      local y = torch.random(h - self.h + 1)
      im = im:narrow(2, y, self.h)
   end
   if w > self.w then
      local x = torch.random(w - self.w + 1)
      im = im:narrow(3, x, self.w)
   end
   if im:size(1) == 1 then
      im = im:expand(3, im:size(2), im:size(3))
   end
   return im
end

function ImageNetDatasource:centerCrop(im)
   local h, w = im:size(2), im:size(3)
   if h > self.h then
      local y = math.floor((h - self.h)/2+1)
      im = im:narrow(2, y, self.h)
   end
   if w > self.w then
      local x = math.floor((w - self.w)/2+1)
      im = im:narrow(3, x, self.w)
   end
   if im:size(1) == 1 then
      im = im:expand(3, im:size(2), im:size(3))
   end
   return im
end

function ImageNetDatasource:nextBatch(batchSize, set)
   assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
   assert(self.dirs[set] ~= nil, 'Unknown set ' .. set)
   self.output_cpu:resize(batchSize, self.nChannels, self.h, self.w)
   self.labels_cpu:resize(batchSize)
   for i = 1, batchSize do
      local class = torch.random(self.nClasses)
      local idx = torch.random(#self.inds[set][class])
      local impath = paths.concat(self.path, self.dirs[set], self.inds[set][class][idx])
      local im = image.load(impath)
      self.output_cpu[i]:copy(self:randomCrop(im))
      self.labels_cpu[i] = class
   end
   return self:typeResults(self.output_cpu, self.labels_cpu)
end

function ImageNetDatasource:orderedIterator(batchSize, set)
   assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
   assert(self.dirs[set] ~= nil, 'Unknown set ' .. set)
   self.output_cpu:resize(batchSize, self.nChannels, self.h, self.w)
   self.labels_cpu:resize(batchSize)
   local classidx, idx = 1, 1
   return function()
      for iBatch = 1, batchSize do
	 if idx > #self.inds[set][classidx] then
	    classidx = classidx + 1
	    idx = 1
	    if classidx > self.nClasses then
	       if iBatch == 1 then
		  return nil
	       else
		  return self.typeResults(self.output_cpu:narrow(1, 1, iBatch-1),
					  self.labels_cpu:narrow(1, 1, iBatch-1))
	       end
	    end
	 end
	 local impath = paths.concat(self.path, self.dirs[set], self.inds[set][classidx][idx])
	 local im = image.load(impath)
	 self.output_cpu[iBatch]:copy(self:centerCrop(im))
	 self.labels_cpu[iBatch] = classidx
	 idx = idx + 1
      end
      return self:typeResults(self.output_cpu, self.labels_cpu)
   end
end
