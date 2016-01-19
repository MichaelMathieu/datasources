require 'datasources.ucf101'
require 'image'

datasource = UCF101Datasource(5)

batch, label = datasource:nextBatch(8, 'train')
print{batch}

torch.setnumthreads(2)

--[[
image.display({image=batch[1][1], legend=1})
image.display({image=batch[1][2], legend=2})
image.display({image=batch[1][3], legend=3})
image.display({image=batch[1][4], legend=4})
image.display({image=batch[1][5], legend=5})

image.display({image=batch[2][1], legend=1})
image.display({image=batch[2][2], legend=2})
--]]


require 'datasources.thread'
require 'cutorch'
datasource = ThreadedDatasource(
   function()
      require 'datasources.ucf101'
      require 'datasources.augment'
      return AugmentDatasource(UCF101Datasource(5), {crop={32,32}})
   end, {nDonkeys=3})
datasource:cuda()

timer = torch.Timer()
for i = 1, 10 do
   batch, label = datasource:nextBatch(4, 'train')
   print{batch}
end
print(timer:time())

--[[
i = 0
for batch, label in datasource:orderedIterator(16, 'test') do
   i = i + 1
   if i == 100 then
      break
   end
end
--]]

print("ok")