#!/usr/bin/ruby
require "neural.rb"

data = []
open("classification.txt") do |f|
  while line = f.gets
    x1, x2, t = line.split
    data << [[x1.to_f, x2.to_f], [t.to_f]]
  end
end

# units
in_units = [Unit.new("x1"), Unit.new("x2")]
bias = [BiasUnit.new("1")]
hiddenunits = [TanhUnit.new("z1"), TanhUnit.new("z2"), TanhUnit.new("z3"), TanhUnit.new("z4"), TanhUnit.new("z5"), TanhUnit.new("z6")]
out_unit = [SigUnit.new("y1")]

# network
network = Network.new
network.in  = in_units
network.link in_units + bias, hiddenunits
network.link hiddenunits + bias, out_unit
network.out = out_unit

eta = 0.1
sum_e = 999999
1000.times do |tau|
  s = 0
  data.each do |d|
    s += network.sum_of_squares_error(d[0], d[1])
  end
  puts "sum of errors: #{tau} => #{s}"
  break if s > sum_e
  sum_e = s

  data.sort{rand}.each do |d|
    grad = network.gradient_E_backward(d[0], d[1])
    network.weights.descent eta, grad
  end
end
network.weights.dump
