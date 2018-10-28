# Description

PolyNN is a neural network library, made from scratch. Unlike classical neural networks, they are able to learn polynomial functions.

# Applications

**logic** My first intent was to allow neural networks to perform fuzzy logic. This requires neural networks to compute polynomials of degree 2.

**Polynomial regression** The most straightforward application is polynomial regression (any number of variables, any degree). The following figure shows (in red) the regression performed by our model given the data ploted in blue :

![ScreenShot](/screenshots/regression_with_noise.png)

# Known limitations

Sometimes, the learning process diverges violently. Making the learning process more stable is a work in progress.

Sofar, only simple (defined componentwise and equal on each component) threshold functions are usable by PolyNN. This is a work in progress.

# Credits
Thanks to Michael Nielsen whose [book](http://neuralnetworksanddeeplearning.com/index.html) gave me great insight about the fundamentals of machine learning.


# License

GNU General Public License v3.0
