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
hiddenunits = (1..6).map{|i| TanhUnit.new("z1#{i}")}
hiddenunits2 = (1..6).map{|i| TanhUnit.new("z2#{i}")}
out_unit = [SigUnit.new("y1")]

# network
network = Network.new(:error_func=>ErrorFunction::CrossEntropy, :code_generate=>true)
network.in  = in_units
network.link in_units, hiddenunits
network.link hiddenunits, hiddenunits2
network.link hiddenunits2, out_unit
network.out = out_unit

eta = 0.1
sum_e = 999999
1000.times do |tau|
=begin
  s = 0
  data.each do |d|
    s += network.error_function(d[0], d[1])
  end
  puts "sum of errors: #{tau} => #{s}"
  break if s > sum_e
  sum_e = s
=end
  data.sort{rand}.each do |d|
    grad = network.gradient_E(d[0], d[1])
    network.weights.descent eta, grad
  end
end
network.weights.dump
