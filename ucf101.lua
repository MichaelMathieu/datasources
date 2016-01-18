require 'torch'
require 'io'
require 'paths'
require 'thffmpeg'
require 'datasources.datasource'

local UCF101Datasource, parent = torch.class('UCF101Datasource', 'ClassDatasource')

function UCF101Datasource:__init(nInputFrames)
   parent.__init(self)
   self.datapath = '/scratch/datasets/ucf101/UCF-101'
   local setfiles = {train = 'trainlist01.txt', test = 'testlist01.txt'}
   assert(paths.dirp(self.datapath), 'Path ' .. self.datapath .. ' does not exist')
   local classes = paths.dir(self.datapath)
   self.classes = {}
   self.sets = {train = {}, test = {}}
   for _, set in pairs{'train', 'test'} do
      local f = io.open(paths.concat(self.datapath, setfiles[set]), 'r')
      assert(f ~= nil, 'File ' .. paths.concat(self.datapath, setfiles[set]) .. ' not found.')
      for line in f:lines() do
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
   self.nInputFrames = nInputFrames
   self.nChannels, self.nClasses = 3, 101
   self.h, self.w = 240, 320
   self.thffmpeg = THFFmpeg()
end

function UCF101Datasource:nextBatch(batchSize, set)
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
	 if (self.thffmpeg:open(filepath)) then
	    if self.nbframes[filepath] == nil then
	       self.nbframes[filepath] = self.thffmpeg:length()
	    end
	    local nframes = self.nbframes[filepath]
	    if nframes >= self.nInputFrames then
	       self.labels_cpu[i] = iclass
	       local istart = torch.random(nframes - self.nInputFrames + 1)
	       self.thffmpeg:seek(istart-1)
	       for j = 1, self.nInputFrames do
		  self.thffmpeg:next_frame(self.output_cpu[i][j])
	       end
	       done = true
	    end
	 end
      end
   end
   self.thffmpeg:close()
   return self.output_cpu, self.labels_cpu
end

function UCF101Datasource:nextIteratedBatch(batchSize, set, idx)
   error("Not implemented")
end
