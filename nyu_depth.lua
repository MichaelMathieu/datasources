--[[ params:
   nInputFrames
   frameRate [1]
--]]

require 'datasources.datasource'
require 'mattorch'
require 'paths'
require 'image'

local NYUDepthDatasource, parent = torch.class('NYUDepthDatasource', 'ClassDatasource')

function NYUDepthDatasource:__init(params)
   parent.__init(self)
   self.datapath = '/scratch/datasets/nyudepth/nyudepth_processed'
   assert(paths.dirp(self.datapath))
   self.h, self.w = 427, 561
   self.nChannels = 4
   self.nInputFrames = params.nInputFrames
   self.frameRate = params.frameRate or 1
   self.vids = {}
   for _, v in pairs(paths.dir(self.datapath)) do
      if v:sub(1,1) ~= '.' then
	 local index = self:buildIndex(paths.concat(self.datapath, v))
	 --TODO: save the indices
	 if #index >= self.nInputFrames then
	    self.vids[1+#self.vids] = {v, index}
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
   assert(set == 'train', 'TODO')
   self.output_cpu:resize(batchSize, self.nInputFrames, self.nChannels, self.h, self.w)
   self.labels_cpu:resize(batchSize):zero() --TODO
   for i = 1, batchSize do
      local done = false
      while not done do
	 done = true
	 local vid_idx = torch.random(#self.vids)
	 local video_path, video_ind = unpack(self.vids[vid_idx])
	 local frame_idx = torch.random(#video_ind - self.frameRate*(self.nInputFrames-1))
	 for j = 1, self.nInputFrames do
	    local frame = mattorch.load(paths.concat(self.datapath, video_path, video_ind[frame_idx+(j-1)*self.frameRate]))
	    local frame_depth = frame.imgDepthFilled
	    if frame_depth:eq(0):any() then
	       done = false
	    else
	       local frame_rgb = frame.imgRgb
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