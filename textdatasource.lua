local TextDatasource = torch.class('TextDatasource')

function TextDatasource:__init()
   self.tensortype = torch.getdefaulttensortype()   
   self.output_cpu = torch.LongTensor()
end

function TextDatasource:typeResults(output)
   if self.tensortype == 'torch.CudaTensor' then
      self.output_gpu:resize(output:size()):copy(output)
      return self.output_gpu
   else
      return output
   end
end

function TextDatasource:type(typ)
   self.tensortype = typ
   if typ == 'torch.CudaTensor' then
      self.output_gpu = torch.CudaTensor()
   else
      self.output_cpu = self.output_cpu:type(typ)
      self.output_gpu = nil
      collectgarbage()
   end
end

function TextDatasource:cuda()
   self:type('torch.CudaTensor')
end

function TextDatasource:float()
   self:type('torch.FloatTensor')
end

function TextDatasource:double()
   self:type('torch.DoubleTensor')
end