# course_work
getting data from Google Trands and making a data analysis; code for music app and song analysis

Song popularity prediction application based on song features analysis -------------------------------------------------------------
Authors: ----------------------------------------------------------------------------------------------------------------------------
https://github.com/juliia5m 

https://github.com/Vlada2909

Statistical analysis of audio characteristics is widely spread now. Many people use it for various purposes: from sorting songs by genre(based on some audio features) to analyzing reasons of popularity of songs. On the other hand, it has not been studied well how music popularity can be defined, what are its characteristics, and whether it can be 'qualitatively' predicted. Since in these days, a large amount of multimedia content is delivered to users through various media and platforms, popularity prediction has become more practical. Our work is a good contribution to this problem.

Extraction of features is a very important part in analyzing and finding relations between different things. Multimedia content items' popularity has been always considered important, because it plays a crucial role in dealing with various issues of content management such as recommendation, search retrieval and many others. 

The main goal of our course work was to create a kivy-application which takes as input a 'wav' audio file and 'txt' file with text of the song and shows a user an output: counted features for their song and prediction about its popularity, based on features' level. For instance, high danceability level can cause high popularity of a song if most people love such songs now or its low level can show composer that they should choose different audience for their creation.

Let's record what exactly we will be doing during our research work:

        1.Analyse existing approaches and decide which audio characteristics we need in order to build our model.
        
        2.Choose an appropriate dataset. As we will analyze audio features, we need data that can provide these features or will be suitable in order to count them.
        3.Clean dataset. Leave only these songs that we can work with.
        4.Extract data from GoogleTrands in order to analyze performer's popularity before and after song release.
        5.Bulid graphics and use received results so that they can have an impact on popularity result for a song;
        6.Deeply analyze features of every song. Create graphics for main counted characteristics in order to see which of them have the most significant influence on a result. 
        7.Deeply analyze characteristics of all songs in general: whether there is a connection between them, if we can notice difference between the number of songs in one genre and in others or whether there is a song in the dataset that shows different results than other songs. Decide whether we should use it in model or not. Is it an exception or rather a problem in the whole research?! 
        8.Check linear regression coefficients for songs that are sorted by sorting parameter: genre in our case. 
        9. If they are close to 1, we build our model. If they are too small or negative, we choose another sorting main parameter and build our model based on it. 
        10. Model returns results that show which features influenced the popularity of the songs the most and how high or low they were. 
        12.Then study the kivy documentation. 
        13. Write the kivy-application with an interface, which accepts a song and a 'txt' file from a user, applies our model on it and returns a counted features and prediction of popularity of a given song based on them. 
        

 
