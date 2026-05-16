This is Linear regression model from scratch. I used AI tools to generate code but I have fully deeply understadning for code. The file is completly human written. 

The first requirement for linear regression model is that there should be linear relationship between features. To find linear relationships we use formula

   **$$Y_{pred} = WX + b$$**
           
where x in input feature and w is weight of feauter. We also one bias term allowing model to acurately fit data that does pass through origin. 
Then we find differenc, known as error,  between acutal data and predicted data. Formula for difference is 

   **$$\text{Loss}_{\text{MSE}} = \frac{1}{2n} \sum_{i=1}^{n} (Y_{pred}^{(i)} - Y_{actual}^{(i)})^2$$**

There are no of reasons for square. One of them is to convert negative error into positive. Then we have another term 1/2. We use this term to cancel square when we calculate derivative .  

   **Derivative:**

We calculate derivative to find slope. It tells us wether to increase weights or decrease to get best results. Large the value of derivative, farther we are from global minima, so we need to decrease value. To update the model we use chain rule to find partial derivatives. 

   **$$\frac{\partial \text{Loss}}{\partial W} = -\frac{2}{n} \sum_{i=1}^{n} X^{(i)} (Y_{actual}^{(i)} - Y_{pred}^{(i)})$$**

For bias 

   **$$\frac{\partial \text{Loss}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (Y_{actual}^{(i)} - Y_{pred}^{(i)})$$** 

Once we have these weights, we update parameters using learning rate, for W it is represented by:

   **$$W_{new} = W_{old} - \alpha \cdot \frac{\partial \text{Loss}}{\partial W}$$**  alpha is learning rate

And for b it is represented by :

   **$$b_{new} = b_{old} - \alpha \cdot \frac{\partial \text{Loss}}{\partial b}$$**  alpha is learning rate

**Standardizaton:**

Scaling is a process where we normalize our data. Because if we have two features one has a value 1000 and second one has value 10, so obviously model will give high priority to high value data. We require scaling for those model that use distance or gradient descent to find best weights. Almost for every model training we scale our data to mean 0 and standard deviation 1. 

   **$$x_{scaled} = \frac{x - \mu}{\sigma + \epsilon}$$**

where X is features , mu is mean. delta is sign of standard deviation  and epsilion. We use epsilion to avoid NAN, becuase if SD becomes 0 then this will not work without epsilon.

**Loss Functions:**

In MSE, we minimize MSE to maximize likelihood solution of the model as noise follows gaussian distribuion, beel shape curve. If after training our model follows gaussian distribuiton then official our model is ok, all those things are noise. In same way for MAE, it follows liplace distribution, sharp peak and heavy tail.
MSE formula is :

   **$$\text{Loss}_{\text{MSE}} = \frac{1}{2n} \sum_{i=1}^{n} (Y_{pred}^{(i)} - Y_{actual}^{(i)})^2$$**

MAE foumula is :

   **$$\text{Loss}_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |Y_{pred}^{(i)} - Y_{actual}^{(i)}|$$**






   
