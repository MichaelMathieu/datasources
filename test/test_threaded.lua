require 'datasources.thread'
require 'cutorch'

datasource = ThreadedDatasource(function()
				   require 'datasources.svhn'
				   return SVHNDatasource()
				end, {nDonkeys = 2})

require 'image'
batch, labels = datasource:nextBatch(3, 'train')
image.display({image=batch, zoom=3})
print(labels)
batch, labels = datasource:nextBatch(3, 'train')
image.display({image=batch, zoom=3})
print(labels)
batch, labels = datasource:nextBatch(3, 'train')
image.display({image=batch, zoom=3})
print(labels)
batch, labels = datasource:nextBatch(2, 'test')
image.display({image=batch, zoom=3})
print(labels)

local i = 0
for batch, labels in datasource:orderedIterator(4, 'test') do
   image.display({image=batch, zoom=3})
   print(labels)
   if i > 2 then
      break
   end
   i = i + 1
end

batch, labels = datasource:nextBatch(4, 'test')
image.display({image=batch, zoom=3})
print(labels)

print ("---------------------------")

datasource = ThreadedDatasource(function()
				   require 'datasources.ucf101'
				   return UCF101Datasource(4)
				end, {nDonkeys = 3})
datasource:cuda()

batch, labels = datasource:nextBatch(3, 'train')
batch = batch:view(-1, batch:size(3), batch:size(4), batch:size(5))
image.display({image=batch})
print(labels)
batch, labels = datasource:nextBatch(3, 'train')
batch = batch:view(-1, batch:size(3), batch:size(4), batch:size(5))
image.display({image=batch})
print(labels)
batch, labels = datasource:nextBatch(2, 'test')
batch = batch:view(-1, batch:size(3), batch:size(4), batch:size(5))
image.display({image=batch})
print(labels)

local i = 0
for batch, labels in datasource:orderedIterator(4, 'test') do
   batch = batch:view(-1, batch:size(3), batch:size(4), batch:size(5))
   image.display({image=batch})
   print(labels)
   if i > 2 then
      break
   end
   i = i + 1
end

batch, labels = datasource:nextBatch(4, 'test')
batch = batch:view(-1, batch:size(3), batch:size(4), batch:size(5))
image.display({image=batch})
print(labels)
