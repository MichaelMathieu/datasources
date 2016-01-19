require 'datasources.augment'
require 'datasources.svhn'
require 'datasources.mnist'

datasource = AugmentDatasource(SVHNDatasource(), {flip = 0,
						  crop = {25,20},
						  scaleup = 1.2,
						  rotate = 1})
datasource = SVHNDatasource()

require 'image'

batch, label = datasource:nextBatch(8, 'train')
image.display({image=batch, zoom=3, legend=1})
print(label)
batch, label = datasource:nextBatch(8, 'train')
image.display({image=batch, zoom=3, legend=2})
print(label)
batch, label = datasource:nextBatch(8, 'train')
image.display({image=batch, zoom=3, legend=3})
print(label)

for batch, label in datasource:orderedIterator(6, 'train') do
   image.display({image=batch, zoom=3})
   break
end