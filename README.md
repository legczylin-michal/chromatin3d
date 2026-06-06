# Dataset creation

Use juicer tools to extract .hic file:
```bash
java -jar juicer_tools.jar dump observed NONE <path/to/input.hic> <chrA> <chrA> BP <binsize> <path/to/output.txt>
```
for example:
```bash
java -jar juicer_tools.jar dump observed NONE ENCFF216QQM/ENCFF216QQM.hic chr3 chr3 BP 5000 ENCFF216QQM/ENCFF216QQM_chr3_5kbp.txt
```
Then in python use this function accordingly to cut the extracted file into Hi-C matrices:
```python
cut_hic(path, binsize, number_of_samples)
```
For example:
```python
cut_hic("ENCFF216QQM/ENCFF216QQM_chr3_5kbp.txt", 5000, 100)
```
Then use Dataset and DataLoader from torch to easily access training:
```python
dataset = HiCDataset(path_to_folder)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
```
For example:
```python
dataset = HiCDataset("ENCFF216QQM/ENCFF216QQM_chr3_5kbp/")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
```

MSVC required for torchsort