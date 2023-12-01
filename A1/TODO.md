Qua possiamo mettere i TODO un po' più verbosi o le note/idee che ci vengono per non perderle

* We are not adding '<pad>' to GloVe, should we do it? -> non necessariamente per ora, visto che prima facciamo parola -> embedding e poi paddiamo in my_collate
* Aggiungere gli OOV del training a GloVe/vocabulary -> <done>
* CEloss -> dobbiamo escludere anche i punctuation? -> il tutor ha detto di no sul forum, top
* la F1Score dobbiamo calcolarla anche sul training come stiamo facendo? Perchè la descrive come una evaluation metric, quindi forse serve solo per validation and test
* Weird '\/' tokens -> Virtuale: