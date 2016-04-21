require 'datasources.pentreebank_char'

datasource = PTBCharDatasource()

x = datasource:nextBatch(42, 'train')

for x in datasource:orderedIterator(4,'valid') do
   print{x}
end