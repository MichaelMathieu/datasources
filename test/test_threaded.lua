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