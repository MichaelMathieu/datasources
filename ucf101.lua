--[[
   params:
     nInputFrames
     minimumMotion [nil]
     frameRate
--]]

require 'torch'
require 'io'
require 'paths'
require 'thffmpeg'
require 'math'
require 'datasources.datasource'

local UCF101Datasource, parent = torch.class('UCF101Datasource', 'ClassDatasource')

function UCF101Datasource:__init(params)
   parent.__init(self)
   assert(params.nInputFrames ~= nil, "UCF101Dataset: must specify nInputFrames")
   self.datapath = params.datapath or '/scratch/datasets/ucf101/UCF-101'
   local setfiles = {train = 'trainlist01.txt', test = 'testlist01.txt'}
   assert(paths.dirp(self.datapath), 'Path ' .. self.datapath .. ' does not exist')
   local classes = paths.dir(self.datapath)
   self.classes = {}
   self.sets = {train = {}, test = {}}
   for _, set in pairs{'train', 'test'} do
      local f = io.open(paths.concat(self.datapath, setfiles[set]), 'r')
      assert(f ~= nil, 'File ' .. paths.concat(self.datapath, setfiles[set]) .. ' not found.')
      for line in f:lines() do
	 if string.byte(line:sub(-1,-1)) == 13 then
	    --remove the windows carriage return
	    line = line:sub(1,-2)
	 end
	 local filename, class
	 if set == 'train' then
	    filename = line:sub(1, line:find(' ')-1)
	    classidx = tonumber(line:sub(line:find(' ')+1, -1))
	    class = filename:sub(1, filename:find('/')-1)
	    self.classes[classidx] = class
	 else
	    filename = line
	    class = filename:sub(1, filename:find('/')-1)
	 end
	 local avifile = filename:sub(filename:find('/')+1,-1)
	 if self.sets[set][class] == nil then
	    self.sets[set][class] = {}
	 end
	 table.insert(self.sets[set][class], avifile)
      end
      f:close()
      local n = 0
      for _, _ in pairs(self.sets[set]) do
	 n = n + 1
      end
      assert(n == 101)
   end
   self.nbframes = {}
   assert(#self.classes == 101)
   self.nInputFrames = params.nInputFrames
   self.frameRate = params.frameRate or 1
   self.minimumMotion = params.minimumMotion
   assert((self.minimumMotion == nil) or (self.minimumMotion > 0))
   self.nChannels, self.nClasses = 3, 101
   self.h, self.w = 240, 320
   self.thffmpeg = THFFmpeg()
end

function UCF101Datasource:testEnoughMotion(frame1, frame2)
   if self.minimumMotion == nil then
      return true
   else
      return (frame1 - frame2):norm() > math.sqrt(self.minimumMotion * frame1:nElement())
   end
end

function UCF101Datasource:nextBatch(batchSize, set)
   assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
   assert(self.sets[set] ~= nil, 'Unknown set ' .. set)
   self.output_cpu:resize(batchSize, self.nInputFrames, self.nChannels, self.h, self.w)
   self.labels_cpu:resize(batchSize)
   for i = 1, batchSize do
      local done = false
      while not done do
	 local iclass = torch.random(self.nClasses)
	 local class = self.classes[iclass]
	 local idx = torch.random(#self.sets[set][class])
	 local filepath = paths.concat(self.datapath, class, self.sets[set][class][idx])
	 local result = self.thffmpeg:open(filepath)
	 if result then
	    if self.nbframes[filepath] == nil then
	       self.nbframes[filepath] = self.thffmpeg:length()
	    end
	    local nframes = self.nbframes[filepath]
	    done = true
	    if nframes >= (self.nInputFrames-1)*self.frameRate+1 then
	       self.labels_cpu[i] = iclass
	       local istart = torch.random(nframes - (self.nInputFrames-1)*self.frameRate)
	       self.thffmpeg:seek(istart-1)
	       for j = 1, self.nInputFrames do
		  local res = self.thffmpeg:next_frame(self.output_cpu[i][j])
		  if res == nil then done = false end
		  if j ~= self.nInputFrames then
		     for k = 1, self.frameRate-1 do
			done = done and self.thffmpeg:skip_frame()
		     end
		  end
	       end
	       if self.nInputFrames > 1 then
		  done = done and self:testEnoughMotion(self.output_cpu[i][-2], self.output_cpu[i][-1])
	       end
	    end
	 else
	    print("can't open", i, threadid_t, filepath)
	 end
	 if not done then print("There was a problem with video " .. filepath) end
      end
   end
   self.thffmpeg:close()
   self.output_cpu:mul(2/255):add(-1)
   return self:typeResults(self.output_cpu, self.labels_cpu)
end

function UCF101Datasource:orderedIterator(batchSize, set, extraargs)
   -- if extraargs.ucf101.only_one_sample_per_video == 'first' or 'random'
   --  then it returns only one set of frames per video
   assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
   assert(self.sets[set] ~= nil, 'Unknown set ' .. set)
   local extraargs = extraargs or {}
   local extraargs = extraargs.ucf101 or {}
   local onesamplepervid = extraargs.only_one_sample_per_video
   local class_idx = 1
   local video_idx = 1
   local frame_idx = 1
   local thffmpeg2 = THFFmpeg()
   return function()
      self.output_cpu:resize(batchSize, self.nInputFrames, self.nChannels,
			     self.h, self.w)
      self.labels_cpu:resize(batchSize)
      for i = 1, batchSize do
	 local done = false
	 while not done do
	    if class_idx > self.nClasses then
	       return nil
	    end
	    local class = self.classes[class_idx]
	    local filepath = paths.concat(self.datapath, class, self.sets[set][class][video_idx])
	    local goodvid = true
	    if frame_idx == 1 then
	       goodvid = thffmpeg2:open(filepath)
	       if onesamplepervid == 'random' then
		  assert('TODO: seek(random(len-nFrames))')
	       end
	    end
	    if goodvid then
	       self.labels_cpu[i] = class_idx
	       for j = 1, self.nInputFrames do
		  if not thffmpeg2:next_frame(self.output_cpu[i][j]) then
		     done, goodvid = false, false
		     break
		  end
		  if j ~= self.nInputFrames then
		     for k = 1, self.frameRate-1 do
			if not thffmpeg2:skip_frame() then
			   done, goodvid = false, false
			   break
			end
		     end
		  end
	       end
	       done = true
	       frame_idx = frame_idx + (self.nInputFrames-1)*self.frameRate+1
	    end

	    if not goodvid then
	       video_idx = video_idx + 1
	       if video_idx > #self.sets[set][class] then
		  class_idx = class_idx + 1
		  video_idx = 1
		  if class_idx > self.nClasses then
		     thffmpeg2:close()
		     return nil
		  end
	       end
	       frame_idx = 1
	    elseif onesamplepervid then
	       video_idx = video_idx + 1
	       if video_idx > #self.sets[set][class] then
		  class_idx = class_idx + 1
		  video_idx = 1
	       end
	       frame_idx = 1
	    end
	 end
      end
      self.output_cpu:mul(2/255):add(-1)
      return self:typeResults(self.output_cpu, self.labels_cpu)
   end
end

function UCF101Datasource:orderedVideoIterator(batchSize, set, extraargs)
   extraargs = extraargs or {}
   extraargs.ucf101 = extraargs.ucf101 or {}
   extraargs.ucf101.only_one_sample_per_video = 'first'
   return self:orderedIterator(batchSize, set, extraargs)
end
