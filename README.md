# FutureMakers
This repository contains all my reflections from the MIT FutureMakers program.


# Day 1 - Welcome To SureStart:

Response propmt: 
Create a section in your README or wiki called Responses to add reflections or
answers for each activity. Remember to mark each response with the associated date.
For Day 1’s reflection, reflect on what you hope to learn in this program


It was so nice to see everyone today! I loved getting to see my mentor and the other students in my group. The talk hosted by Ms. Boucaud was great as well (I loved her energy). Today was awesome, and I can't wait to start learning about ML.


# Day 2 - Become A Leader: 
Response Prompt: For Day 2’s reflection, reflect on what you learned in Dr. David Kong’s leadership
seminar about yourself, the world, and what are your unique contributions to your
local and global community. 

I absolutely loved hearing everyone's stories. It was incredibly inspiring and remarkable to watch the videos and then hear from my peers. I was a bit nervous about telling my story, but I gained confidence in it after hearing how the story should be formulated. I can't wait to learn new things tomorrow!

# Day 3 - Introduction to ML and Scikit Learn: 
Response Prompt: Post short answers to the following questions to the Responses section of your README/wiki. What is the difference between supervised and unsupervised learning? Describe why the following statement is FALSE: Scikit-Learn has the power to visualize data without a Graphviz, Pandas, or other data
analysis libraries.


In supervised learning, a program is trained on pre-defined sets of examples. This is in hopes of improving the model's ability to make correct predictions when present with new data. However, in unsupervised learning, a program is given data and has to find relationships/patterns inside the data (it is not used to make future predictions).

Statement: Scikit-Learn has the power to visualize data without a Graphviz, Pandas, or other data analysis libraries. This statement is false because it Scikit-Learn uses all these libraries to handle the data that it will then graph.

# Day 4 - Deep Learning (IBM Dataset): 

Response Prompt: Think about a real-world problem and see if you can find a dataset that has the characteristics of the data of that problem. Then, think about the deep learning algorithm that you would likely use to develop a solution to it. Outline why you picked a particular approach.

The issue I want to focus on is the fact that facial recognition software has historically underperformed on the faces of Black and Brown people. After doing some research, I discovered that IBM released a data set of over 1 million people to promote fairness and increase accuracy in facial recognition technology. (https://www.ibm.com/blogs/research/2019/01/diversity-in-faces/) (https://www.research.ibm.com/artificial-intelligence/trusted-ai/)

For this dataset, I would likely use a convolutional neural network. This model would work best since it (a) works with images, (b) isn't prone to overfitting, (c) isn't heavily computational (I do not have an incredibly large amount of memory to work with), and (d) is relatively fast. I could also use a recurrent neural network, but I am not sure how much memory it would need for its operation.

# Day 5: 
Weekend (N/A)

# Day 6:
Weekend (N/A)

# Day 7 - CNNS, DATA & MACHINE LEARNING:
Response Prompt: 
1) What are “Tensors” and what are they used for in Machine Learning? 
2) What did you notice about the computations that you ran in the TensorFlow programs (i.e. interactive models) in the tutorial?

A tensor is a mathematical representation of something that has magnitude and multiple directions

I noticed that the computations weren't entirely straightforward. The information being printed wasn't in plain English (before this, I usually printed strings or numbers, and I'd know exactly what to expect from it) and it took me a while to orient myself and understand exactly what was being printed. I expect this to get easier with practice, especially since this was my first time working on these types of computations.


# Day 8 - What Are Neural Networks (NN)?:

Response Prompt: N/A

The overview for today was very informative! I felt a little lost when I realized that I had to build a model on my own. I conceptually understand how the different pieces work, but I am not as strong on the actual code. I found someone's code (credit to user Madhav Mathur https://www.kaggle.com/madz2000/sarcasm-detection-with-glove-word2vec-83-accuracy) and used their model. It took me a while to look up each line and what it meant. I still do not completely understand, but I believe this exercise is good to start with building models from scratch.

Update (7/20)

I went through my code and fixed all the errors. In the end, the model was overfitting by a pretty extreme amount (it gained 99% accuracy on the training data but had an 80% accuracy on the testing data). In the future, I will combat the overfitting by adding dropout layers and regularization. 

# Day 9 - Intro To Convolutional Neural Networks (CNNS):
Response Prompt: N/A

I was able to understand the daily overview, but the article was a bit more complicated (https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/). I found myself rereading sentences and entire passages in order to grasp an understanding. (I understand that this is part of the learning process, however.) I do have some more work to do on my CNN model, as I ran into errors when trying to upload a new dataset to the model. (Tomorrow, I need to ask the proper way to upload a file to the ML model. As for the article, I made sure to take good notes, as I know that I will be referencing them moving forward. I can't wait for tomorrow's activities!

# Day 10 - Algorithmic Bias And Data Sets:

Response Prompt: 
Play “Survival of the Best Fit" (https://www.survivalofthebestfit.com/) to learn more about how AI might impact human resources and hiring processes in different fields. When you have completed this game, answer the following questions: 

1) How do you think Machine Learning or AI concepts were utilized in the design of this game?

2) Can you give a real-world example of a biased machine learning model, and share your ideas on how you make this model more fair, inclusive, and equitable? Please reflect on why you selected this specific biased model.

When playing "Survival of the Best Fit", I noticed that AI is incredibly reliant on existing datasets, and that it depends heavily on communication between the producers and users of the algorithms. I also began to think about how bad the real-world effects of ML can quickly become if left unchecked or unquestioned.

A real-world example of a biased ML model would be "This Person Does Not Exist" (https://thispersondoesnotexist.com/). I like to draw in my spare time and I used this site to practice faces, as it generates a face by using GANS. I stopped using the site because none of the faces looked like mine. Most of the faces were of white and male "people", and it was incredibly rare to see a black woman. A way to fix this would be to train a model using diverse datasets. This would expose the model to different types of people, so it can create different types of "people". Another factor to help would be having a diverse team working on the model. It was originally created by one person - Phillip Wang. By having people of different backgrounds working on this model, an issue like this would have been caught and fixed early on.

# Day 11 - Neural Network Layers and Continue Practice With MNIST Digits:

Response Prompt: Succinctly list the differences between a Convolutional Neural Network and a Fully Connected Neural Network. Discuss layers and their role, and applications of each of the two types of architectures.

In a Fully Connected Neural Network, each neuron from one layer is connected to each neuron in the next layer. CNNs, however, consist of a sequence of unique layers (convolution, pooling, etc.). This difference makes Fully Connected Neural Networks broader and CNNs to be used for more specific applications. Another difference is that CNNs are typically used for image inputs (Fully Connected Neural Networks are more flexible.)

There will be an input layer in a Fully Connected Neural network, x amount of hidden layers, and an output layer. Each hidden layer and the output layer will have an activation function assigned to them. The hidden layers take an input, do some calculations, and pass it on to the next layer. The output layer is responsible for feeding an output, and the input layer is responsible for taking something in for the model to process.

In a CNN, there will be an Input, Convolution, Pooling, and Fully Connected layer. The input layer is responsible for taking in/starting to process the input (likely an image). The convolution layer is responsible for scanning the input's dimensions and outputting a feature map. The pooling layer is responsible for downsampling the feature map. The fully connected layer is responsible for flattening the pooling layer's output and optimizing the objective. At the end of the fully connected layer is an output (something the machine spits out).

# Day 12: 
Weekend (N/A)

# Day 13: 
Weekend (N/A)

# Day 14 - Functions And Predictions: 
Response Prompt: N/A

Today was a great day! I ran into a few bugs with my housing prices model, but I was able to recover and overcome the errors with relatively little time lost. It was great to see my model gain accuracy as the number of epochs increased and itt was cool to walk through different parameters for the three different models. I learned a lot about overfitting, and I'll be able to implement strategies like adding dropout layers and/or regularization in future projects!

# Day 15 - Activation Functions: 
Response Prompt: Write a reflection piece on the advantages of the Rectified Linear activation function, along with one use case.

Today, we learned about the different activation functions used for hidden layers and the output layer of the function. The rectified linear unit (ReLU) function has a lot of advantages over the Sigmoid and Tanh functions, as it

- requires less computational power
- can output a true zero value (this helps to accelerate the learning and simplifies the model)
- makes the neural network easier to optimize (it's easier when the behaviour is linear)
- successfully trains deep multi-layer neural networks during backpropagation using a nonlinear activation function.

A use of the ReLU activation function is for overcoming the vanishing gradient problem. This is desirable, as models will learn faster and have a better performance.

# Day 16 - Ethics-Driven ML Practice: 

Response Prompt: N/A

When I ran today's model, I came across  a major issue - runtime. I was using Google Colab, and the model was running for over 2 hours before I stopped it. My older brother informed me that I needed to use a GPU, which exponentially improved the runtime. My model gained a 95% accuracy which was surprising, as I didn't expect it to be good at this type of classification. (I imagined this gender classification to be more abstract than what it really was.) In the coming days, I want to do more customization with the model - i.e changing the functions, adding dropout layers, etc.

# Day 17 - Image Classification And ML: 

Response Prompt: N/A

Today was a bit disappointing. I severely underestimated the amount of time it took to run the model each time, so I wasn't able to make all the changes that I wished to make. (I was able to change 50 epochs to 30 epochs, but I won't be able to check how it affected the accuracy for over an hour). Tomorrow, I'll be more prepared and start coding earlier in the day.

# Day 18 - Data Overfitting And Regularization:

Response Prompt: Add your observations while changing the loss to regression based functions from the housing prices model to the README.

Good news! I managed my time well and gave myself what I needed to do my code (it also helped that the dataset wasn't excessively large). When I changed from a loss to a regression-based function, I did not observe a large difference in the model. When I changed the loss function from binary cross-entropy to mean squared error, the model accuracy is 50% and the validation accuracy is 47%. With the binary cross-entropy, the accuracy is 52% and the validation accuracy is 47%. I look forward to more practice tomorrow!

# Day 19: 
Weekend (N/A)

# Day 20: 
Weekend (N/A) 

# Day 21: 
Response Prompt: N/A

Today's material was a bit harder to wrap my head around and I will definitely need to do some supplementary research on autoencoders. For the building an autoencoder tutorial, I ran into an error that I wasn't able to solve on my own. My plan is to ask about the error in tomorrow's mentor check-in meeting and then finish the rest of the tutorial.

Edit: 
I met with my mentor and we were able to solve the issue. I uploaded the fixed code to GitHub and I need to keep a better eye out for detail moving forward!

# Day 22 - Upsampling and Autoencoders:
Response Prompt: N/A

Today's general meeting was very informative! I loved hearing about the different projects that can be completed using affective computing. For the action item today, I wasn't able to upload the Speech Emotion Analyzer project to Github, so I could not experiment with it. I also wasn't able to play with the audio recorder aspect, as the "frames_per_buffer=CHUNK) #buffer" line caused errors. (I haven't worked with this kind of code, and I wasn't able to find an effective solution on StackOverflow, so I will ask about it tomorrow.)

CREDIT TO: https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer

Edit: We went over the errors in the mentor check-in, and I have uploaded related code to the main branch of this repository. (It is a file that collects, records, and displays a graph of a certain sound.) We realized that there was a special process to record and play back sound in Google Colab, and the uploaded file shows the correct way to go about that.

# Day 23 - Natural Language Processing (NLP): 

Response Prompt: Write a reflection piece on the ethical implications of big NLP models such as GPT-2 and add it to your Responses section.

Today's action item was incredibly interesting! I love that we are in the stage of working on real-world applications of machine learning now.
As for large NLP models like GPT-3, an ethical implication is what the model will be used for. If released, people could use it to generate bogus contracts or create scams that look professional and realistic, for example.




