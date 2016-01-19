require 'datasources.mnist'
require 'image'

datasource = MNISTDatasource()

batch, labels = datasource:nextBatch(8, 'train')

image.display(batch)
print(labels)

i = 0
for batch, labels in datasource:orderedIterator(8, 'train') do
   image.display(batch)
   print(labels)
   if i > 2 then
      break
   end
   i = i + 1
end

i = 0
for batch, labels in datasource:orderedIterator(8, 'train') do
   i = i + 1
end
print("ok", i*8)

i = 0
for batch, labels in datasource:orderedIterator(4, 'test') do
   i = i + 1
end
print("ok", i*4)