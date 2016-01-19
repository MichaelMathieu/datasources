require 'datasources.thread'

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
