This is Linear regression model from scratch. I used AI tools to generate code but I have fully deeply understadning for code. The file is completly human written. 

The first requirement for linear regression model is that there should be linear relationship between features. To find linear relationships we use formula

   **Y_pred = Wx + b**
           
where x in input feature and w is weight of feauter. We also one bias term allowing model to acurately fit data that does pass through origin. 
Then we find differenc, known as error,  between acutal data and predicted data. Formula for difference is 

   **Loss = 1/2(Y_pred - Y_actual)^2**

There are no of reasons for square. One of them is to convert negative error into positive. Then we have another term 1/2. We use this term to cancel square when we calculate derivative .  

   **Derivative:**

We calculate derivative to find slope. It tells us wether to increase weights or decrease to get best results. Large the value of derivative, farther we are from global minima, so we need to decrease value. To update the model we use chain rule to find partial derivatives. 

   **derivative_Loss/derivative_w = -2/n(∑{i=1}^n(x(i) (y(i)-y_predicted)**

For bias 

   **derivative_Loss/derivative_b = -2/n(∑{i=1}^n (y(i)-y_predicted)**

Once we have these weights, we update parameters using learning rate, for W it is represented by:

   **W_new = W_old - alpha. (derivative_Loss/derivative_w)**  alpha is learning rate

And for b it is represented by :

   **b_new = b_old - alpha. (derivative_Loss/derivative_w)**  alpha is learning rate

**Standardizaton:**

Scaling is a process where we normalize our data. Because if we have two features one has a value 1000 and second one has value 10, so obviously model will give high priority to high value data. We require scaling for those model that use distance or gradient descent to find best weights. Almost for every model training we scale our data to mean 0 and standard deviation 1. 

   **$$x_{scaled} = \frac{x - \mu}{\sigma + \epsilon}$$**

   
