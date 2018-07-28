# named-entity-recognizer
Implementation of a named-entity recognizer for the English language
A named-entity recognition model is trained with a corpus containing almost 14,000 English sentences.  
  
## Model  
Model (mini model) in the repo is trained using mini data which contains only almost 3,000 English sentences.  
Therefore, it does not perform well enough.  
  
I encourage you to train a new model on your own using the corpus in data/ directory.  
My training lasted almost 10 minutes on a device with 3,1 GHz Intel Core i7 processor.  
Training accuracy was **100%** where test accuracy was **97%**.  
  
## Usage  
$**python3**  ner.py  input-sentence  
  
### Example  
$**cd**  src  
$**python3** ner.py  "Joe Strummer was born in Ankara."  
**->** Named entity recognizer is loaded.  
**->** [('Joe', 'B-PER'), ('Strummer', 'I-PER'), ('was', 'O'), ('born', 'O'), ('in', 'O'), ('Ankara', 'B-LOC'), ('.', 'O')]
  
*P.S. Above example is tested with a model trained on corpus in data/ directory(with almost 14,000 English sentences).*   
  
*Have fun.*
