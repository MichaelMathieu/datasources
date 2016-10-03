--[[ params:
   nInputFrames
   frameRate [1]
   onlyFullDepth [true]
   basepath [/scratch/datasets/nyudepth/nyudepth_processed]
--]]

require 'datasources.datasource'
require 'mattorch'
require 'paths'
require 'image'

local NYUDepthDatasource, parent = torch.class('NYUDepthDatasource', 'ClassDatasource')

function NYUDepthDatasource:__init(params)
   parent.__init(self)
   assert(params ~= nil, 'Must specify parameters')
   self.basepath = params.basepath or '/scratch/datasets/nyudepth/nyudepth_processed'
   self.datapaths = {train = paths.concat(self.basepath, 'train'),
		     test = paths.concat(self.basepath, 'test')}
   assert(paths.dirp(self.datapaths.train))
   assert(paths.dirp(self.datapaths.test))
   self.h, self.w = 427, 561
   self.nChannels = 4
   self.nInputFrames = params.nInputFrames
   assert(self.nInputFrames ~= nil, 'Must specify nInputFrames')
   self.onlyFullDepth = params.onlyFullDepth
   if self.onlyFullDepth == nil then
      self.onlyFullDepth = true
   end
   self.frameRate = params.frameRate or 1
   self.vids = {}
   for _, set in pairs{'train', 'test'} do
      self.vids[set] = {}
      for _, v in pairs(paths.dir(self.datapaths[set])) do
	 if v:sub(1,1) ~= '.' then
	    local index = self:buildIndex(paths.concat(self.datapaths[set], v))
	    --TODO: save the indices
	    if #index >= self.nInputFrames then
	       self.vids[set][1+#self.vids[set]] = {v, index}
	    end
	 end
      end
   end
end

function NYUDepthDatasource:buildIndex(path)
   local files = paths.dir(path)
   local out = {}
   for _, f in ipairs(files) do
      if f:sub(1,1) ~= '.' then
	 out[1+#out] = f
      end
   end
   table.sort(out)
   return out
end

function NYUDepthDatasource:nextBatch(batchSize, set)
   assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
   self.output_cpu:resize(batchSize, self.nInputFrames, self.nChannels, self.h, self.w)
   self.labels_cpu:resize(batchSize):zero() --TODO
   for i = 1, batchSize do
      local done = false
      while not done do
	 done = true
	 local vid_idx = torch.random(#self.vids[set])
	 local video_path, video_ind = unpack(self.vids[set][vid_idx])
	 local frame_idx = torch.random(#video_ind - self.frameRate*(self.nInputFrames-1))
	 for j = 1, self.nInputFrames do
	    local frame = mattorch.load(paths.concat(self.datapaths[set], video_path, video_ind[frame_idx+(j-1)*self.frameRate]))
	    local frame_depth = frame.imgDepthFilled
	    if self.onlyFullDepth and frame_depth:eq(0):any() then
	       done = false
	    else
	       local frame_rgb = frame.imgRgb
	       if frame_rgb == nil then
		  done = false
		  break
	       end
	       self.output_cpu[i][j][{{1,3}}]:copy(frame_rgb:transpose(2,3))
	       self.output_cpu[i][j][4]:copy(frame_depth:t())
	    end
	 end
      end
   end
   self.output_cpu:narrow(3,1,3):mul(2/255):add(-1)
   self.output_cpu:select(3,4):add(-2.5):div(1.5) --approx between -1 and 1
   return self:typeResults(self.output_cpu, self.labels_cpu)
end