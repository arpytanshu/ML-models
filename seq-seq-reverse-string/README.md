A Sequence to Sequence LSTM model that learns to reverse a string.
Very Fancy!!


The toy data was generated using https://github.com/google/seq2seq/blob/master/docs/data.md

```
source text                     target text
p s m v r f i j w t p g f a a   a a f g p t w j i f r v m s p
m y x i u z e v d v l t x c     c x t l v d v e z u i x y m
x o n b y y                     y y b n o x
y r o e v q b x                 x b q v e o r y
```

The Model was trained for sequence lengths upto 15 characters.
The script train.py can be modified to use source sequences of any length, but decoded sequence's accuracy may suffer.

```                    
                    |''''''|                                |'''''''|
source_sequence ==> |encode| ==> encoded representation ==> |decoder| ==> reversed_sequence
    "ABC"           |______|           "%$#"                |_______|           "CBA"

```

The train script generates the data and train a seq2seq model. The model is saved for inference.
The inference model used the weights from the trained model. see run_small() in run.py
```
Sample Run:  
Input sequence:	a b c d e f g h i j k l  
Dcded sequence:	l k j i h g f e d c b a  

Input sequence:	s w a g o v e r  
Dcded sequence:	r e v o g a w s  
  
Input sequence:	a b c d  
Dcded sequence:	d c b a  
```

