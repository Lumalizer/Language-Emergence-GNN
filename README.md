# Language Emergence GNN
 Language emergence project using graph neural networks, comparing graph based representations to image based representations.

 ## Important: the following manual edit to EGG/core/interaction.py is required:

change line 209
```
                    aux_input[k] = _check_cat([x.aux_input[k] for x in interactions])
```

to
```
                try:
                    aux_input[k] = _check_cat([x.aux_input[k] for x in interactions])
                except Exception as e:
                    aux_input[k] = None
```
 
 Examples of running experiments and plotting results in main.py.
