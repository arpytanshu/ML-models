# Character Level Language Model using tf.keras
Corpus : Nietzsche's writings

A Language Model is a predictive model to which given a sequence of entities, will predict the next entity in the sequence.
A Character Level Language model is one which predict the next character, given all previous characters in the sequence.

Text from the corpus (nietzsche's writings) is left almost unprocessed. The bare minimum pre-processnig appled is:
lowercase all charecters, and discard all special characters except comma(,), dot(.) and spaces( ).

### Sample Text:
````
text[:500]
'this is a shareware trial project it is not freeware we need your support to continue west by north by  
jim prentice copyright , jim prentice, brandon, manitoba, canada north of . a magic phrase. spoken,  
mumbled or thought inwardly by thousands of souls venturing northward. an imaginary line, shown only on  
maps and labelled degrees. its presence indicated to highway travellers by road side signs. a division  
of territory as distinct in the mind as any international border. if you have not been nor'
````
### Squences of fixed length from text
````
sentences[0:6] 
['this is a shareware trial project it is ',
 'trial project it is not freeware we need',
 'not freeware we need your support to con',
 ' your support to continue west by north ',
 'tinue west by north by jim prentice copy',
 'by jim prentice copyright , jim prentice']
 ````
### Model
The model itself is pretty simple.
1 LSTM with 128 units + 1 Dense Layer + 1 softmax Activation

* lstm (LSTM)                  (None, 128)               80896     
* dense (Dense)                (None, 29)                3741      
* activation (Activation)      (None, 29)                0         

### Output
After each epoch, we use a seed phrase and let the model predict characters.  
In the beginning, its gibberish. By 29th epoch, the words formed from the characters do make sense, but the sentences are still non-sensical.

## Sample Output:

````
==== Generating Text after Epoch: 0 ====

== diversity: 0.25 ==
generating with seed: " ther withdrawal symptom...now where are  "
the and and the soun the sound sous and and and and and boun the care the tore sithe bithe sound and  
and and and at and the and and and in the pore sime shing he sound and and and the sof the the sared  
and the seand and and the sale and the and the sion the sing and the ind sor and and of and in the  
sing the cound and the sound the sing the southe the pare and sound the saind the sout the sare the
````
  

````
==== Generating Text after Epoch: 14 ====

== diversity: 0.25 ==
generating with seed: " care have a care you cant play withedged "
and the start of the start of the senser the street stranger to the same of the start of the start  
of the startes of the start of the start for the strangers of the started in the start of the starts  
and she to be a lot and stranger to see the control of the ground of the start of the start of the  
starts of the part of the story of the street of the strong of the street stood and shouldered the m  
````

````
==== Generating Text after Epoch: 29 ====

== diversity: 0.25 ==
generating with seed: " d the axe two men were journeying togeth "
er for the door and the starion of the starter of the forther to the continual and the time the words  
of the start. the later was a country of the stared and the stares which had been the stares of the  
sun was so the stares of the starting of the start of the standing of the single of the started of  
the said. the town is a short to the stood and the word and the first have a bearthered and the way  
````
