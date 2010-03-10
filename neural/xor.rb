#!/usr/bin/ruby

require "neural.rb"
OUTPUT_CODE = false

# training data
D = [
  [[0, 0], [0]],
  [[1, 1], [0]],
  [[0, 1], [1]],
  [[1, 0], [1]],
]

# units
in_units = [Unit.new("x1"), Unit.new("x2")]
hiddenunits = [TanhUnit.new("z1"), TanhUnit.new("z2"), TanhUnit.new("z3"), TanhUnit.new("z4")]
out_unit = [SigUnit.new("y1")]

# network
network = Network.new(:error_func=>ErrorFunction::CrossEntropy)
network.in  = in_units
network.link in_units, hiddenunits
network.link hiddenunits, out_unit
network.out = out_unit

t1 = Time.now.to_f
eta = 0.1
sum_e = 999999
10000.times do |tau|
  s = 0
  D.each do |data|
    s += network.error_function(data[0], data[1])
  end
  #puts "sum of errors: #{tau} => #{s}"
  break if s > sum_e
  sum_e = s

  D.sort{rand}.each do |data|
    grad = network.gradient_E(data[0], data[1])
    network.weights.descent eta, grad
  end
end
#network.weights.dump

t2 = Time.now.to_f
puts "#{RUBY_VERSION}(#{RUBY_RELEASE_DATE})[#{RUBY_PLATFORM}] #{((t2-t1)*1000).to_i/1000.0} sec"

#puts "(0, 0) => #{network.apply(0, 0)}, (1, 1) => #{network.apply(1, 1)}"
#puts "(0, 1) => #{network.apply(0, 1)}, (1, 0) => #{network.apply(1, 0)}"

