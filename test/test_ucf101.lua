require 'datasources.ucf101'
require 'image'

datasource = UCF101Datasource(5)

data = datasource:nextBatch(8, 'train')
batch = data[1]
print{batch}

image.display({image=batch[1][1], legend=1})
image.display({image=batch[1][2], legend=2})
image.display({image=batch[1][3], legend=3})
image.display({image=batch[1][4], legend=4})
image.display({image=batch[1][5], legend=5})

image.display({image=batch[2][1], legend=1})
image.display({image=batch[2][2], legend=2})