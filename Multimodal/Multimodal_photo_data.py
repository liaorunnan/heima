from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('Tongyi-DataEngine/SA1B-Dense-Caption', subset_name='default', split='train')
# print(ds[0])