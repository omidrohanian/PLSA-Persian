Author: Mojtaba "Omid" Rohanian

This project uses [Probabilistic Latent Semantic Analysis](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis) to categorize textual data into separate semantic groups. There are 15 random text documents in Persian, each can be uniquely mapped to a separate subject: "15-day weight loss diet plan", "religion" and "chess". 

The program classifies documents into K separate groups (K is manually set to be 3) and prints the most probable words in each category.

Tokenization and Normalization of text was done using the open source NLP package [Hazm](https://pypi.python.org/pypi/hazm/0.4). Stop words were deleted in the preprocessing stage. The Persian stop words were taken from here:

Kazem Taghva, Russell Beckley, Mohammad Sadeh(2003) A List of Farsi Stop
words, ISRI Technical Report No. 2003-01 Information Science Research Institute University of Nevada, Las Vegas

I also borrowed code from a github repo that seems to have been removed for a while. If you happen to know the original source drop me a line and I'll give proper reference. 

There is a documentation included (in Persian) for further clarification.
