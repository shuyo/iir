#!/usr/bin/ruby

require "neural.rb"

# training data ( y = sin(2 PI x) + N(0, 0.3) )
D = [
  [0.000000,  0.349486], [0.111111,  0.830839],
  [0.222222,  1.007332], [0.333333,  0.971507],
  [0.444444,  0.133066], [0.555556,  0.166823],
  [0.666667, -0.848307], [0.777778, -0.445686],
  [0.888889, -0.563567], [1.000000,  0.261502],
]

# units
in_unit = Unit.new("x1")
bias = BiasUnit.new("1")
hiddenunits = [TanhUnit.new("z1"), TanhUnit.new("z2"), TanhUnit.new("z3"), TanhUnit.new("z4")]
out_unit = IdentityUnit.new("y1")

# network
network = Network.new
network.in  = [in_unit]
network.link [in_unit, bias], hiddenunits
network.link hiddenunits + [bias], [out_unit]
network.out = [out_unit]

eta = 0.1
1000.times do |tau|
  error1 = error2 = 0
  D.sort{rand}.each do |data|
    error1 += network.sum_of_squares_error([data[0]], [data[1]])
    grad = network.gradient_E([data[0]], [data[1]])
    network.descent_weights eta, grad
    error2 += network.sum_of_squares_error([data[0]], [data[1]])
  end
  puts "error func: #{error1} => #{error2}"
end
network.weights.dump



=begin
x = 0.0
while x < 1.0
  y = network.apply(x)
  p [x, y]
  x += 0.05
end
=end
