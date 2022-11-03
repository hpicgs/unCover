# XAI Project

See [this](https://stackoverflow.com/a/22269678/17614473) to install
StanfordParser!

## Some stuff

```
                         S                                 
                     ____|_______________________________   
                    VP                                   | 
  __________________|____                                |  
 |                       S                               | 
 |         ______________|____                           |  
 |        |                   VP                         | 
 |        |          _________|_____                     |  
 |        |         |               VP                   | 
 |        |         |     __________|______              |  
 |        NP        |    |                 NP            | 
 |        |         |    |           ______|______       |  
 VB      NNP        TO   VB         JJ           NNS     . 
 |        |         |    |          |             |      |  
Use StanfordParser  to parse     multiple     sentences  . 

    (VB, Use)
        |________________________
        |                        |
(NNP, StanfordParser)         (TO, to)
                                 |
                            (VB, parse)
                           ______|__________      
                          |                 |
                  (JJ, multiple)    (NNS, sentences)
```
