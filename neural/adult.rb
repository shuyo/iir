#!/usr/bin/ruby

require "neural.rb"
#OUTPUT_CODE = true
LOGFILE = "adult.log"

# training data
CLZ = {"<=50K"=>0, ">50K"=>1}
dataset = []
categories = (1..14).map{[]}
open("adult.data") do |f|
  while line = f.gets
    buf = line.chomp.split(',').map{|x| x.strip}
    clz = buf.pop
    break unless clz
    buf.each_with_index do |x, i|
      if x !~ /^[0-9]+$/
        categories[i] << x if !categories[i].include?(x)
      end
    end
    dataset << [buf, CLZ[clz]]
  end
end

dataset.each_with_index do |data, idx|
  vector = []
  data[0].each_with_index do |x, i|
    if categories[i].length > 0
      one_of_k = [0] * categories[i].length
      one_of_k[categories[i].index(x)] = 1
      vector.concat(one_of_k)
    else
      vector << x.to_f
    end
  end
  dataset[idx] = [vector, data[1]]
end

#dataset = dataset[0, 100]


# units
in_units = (1..dataset[0][0].length).map{|i| Unit.new("x#{i}")}
hiddenunits1 = (1..20).map{|i| TanhUnit.new("z#{i}")}
#hiddenunits2 = (1..30).map{|i| TanhUnit.new("w#{i}")}
out_unit = [SigUnit.new("y1")]

# network
network = Network.new(:error_func=>ErrorFunction::CrossEntropy)
network.in  = in_units
network.link in_units, hiddenunits1
network.link hiddenunits1, out_unit
network.out = out_unit

open(LOGFILE, "a") {|f| f.puts "==== start (#{Time.now})" }

max_correct = 0
10.times do |trial|
  network.weights.init_parameters

  100.times do |tau|
    eta = if tau<10 then 0.1 elsif tau<50 then 0.05 elsif tau<100 then 0.01 else 0.005 end
    t = Time.now.to_i
    dataset.sort{rand}.each do |data|
      grad = network.gradient_E(data[0], [data[1]])
      network.weights.descent eta, grad
      #e += network.error_function(data[0], [data[1]])
    end
    puts "#{tau}: #{Time.now.to_i - t}s"
  end

  correct = 0
  dataset.each do |data|
    y = network.apply(*data[0])
    correct += 1 if (data[1]==0 && y[0]<0.5) || (data[1]==1 && y[0]>0.5)
  end

  #log
  log = "#{trial+1}: correct = #{correct}, mistake = #{dataset.length - correct}, rate = #{(10000.0*correct/dataset.length).to_i/100.0}"
  puts log
  open(LOGFILE, "a") do |f|
    f.puts log
    if max_correct < correct
      max_correct = correct
      f.puts network.weights.dump
    end
  end
end
open(LOGFILE, "a") {|f| f.puts "==== end (#{Time.now})" }

