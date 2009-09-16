#!/usr/bin/ruby

require "neural.rb"
#OUTPUT_CODE = true
LOGFILE = "iris.log"

# training data
CLZ = {"Iris-setosa"=>[1,0,0], "Iris-versicolor"=>[0,1,0], "Iris-virginica"=>[0,0,1]}
dataset = []
open("iris.data") do |f|
  while line = f.gets
    buf = line.chomp.split(',')
    clz = buf.pop
    break unless clz
    x = buf.map{|x| x.to_f}
    dataset << [x, CLZ[clz]]
  end
end

def generate_network(network_type)
  # units
  in_units = (1..4).map{|i| Unit.new("x#{i}")}
  hiddenunits1 = (1..6).map{|i| TanhUnit.new("z#{i}")}
  hiddenunits2 = (1..6).map{|i| TanhUnit.new("w#{i}")}
  out_unit = (1..3).map{|i| SoftMaxUnit.new("y#{i}")}

  # network
  network = Network.new(:error_func=>ErrorFunction::SoftMax)
  network.in  = in_units

  name = nil
  case network_type
  when 0
    name = "full link(6)"
    network.link in_units, hiddenunits1
    network.link hiddenunits1, out_unit
  when 1
    name = "full link(12)"
    network.link in_units, hiddenunits1+hiddenunits2
    network.link hiddenunits1+hiddenunits2, out_unit
  when 2
    name = "full link(6+6)"
    network.link in_units, hiddenunits1
    network.link hiddenunits1, hiddenunits2
    network.link hiddenunits2, out_unit
  when 3
    name = "each 2-input-units"
    network.link [in_units[0], in_units[1]], [hiddenunits1[0]]
    network.link [in_units[0], in_units[2]], [hiddenunits1[1]]
    network.link [in_units[0], in_units[3]], [hiddenunits1[2]]
    network.link [in_units[1], in_units[2]], [hiddenunits1[3]]
    network.link [in_units[1], in_units[3]], [hiddenunits1[4]]
    network.link [in_units[2], in_units[3]], [hiddenunits1[5]]
    network.link hiddenunits1[0, 6], out_unit
  end

  network.out = out_unit
  [name, network]
end

N_TRIALS = 100

open(LOGFILE, "a") {|f| f.puts "==== start (#{Time.now})" }

4.times do |network_type|
  name, network = generate_network(network_type)
  max_correct = 0
  sum_correct = 0

  open(LOGFILE, "a") {|f| f.puts "-- #{name}" }
  t0 = Time.now.to_i
  N_TRIALS.times do |trial|
    network.weights.init_parameters 0, 3
    200.times do |tau|
      eta = if tau<10 then 0.1 elsif tau<50 then 0.05 elsif tau<100 then 0.01 else 0.005 end
      dataset.sort{rand}.each do |data|
        grad = network.gradient_E(data[0], data[1])
        network.weights.descent eta, grad
      end
    end

    correct = 0
    dataset.each do |data|
      y = network.apply(*data[0])
      predict = (0..2).max_by{|i| y[i]}
      #puts "y = #{y.map{|x| (x*10000).to_i/10000.0}.inspect}, answer = #{data[1].inspect}"
      correct += 1 if data[1][predict]==1
    end
    sum_correct += correct

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
  open(LOGFILE, "a") do |f|
    f.puts "max of rate = #{(10000*max_correct/dataset.length).to_i/100.0}, average of rate = #{(10000*sum_correct/(dataset.length*N_TRIALS)).to_i/100.0} (#{Time.now.to_i - t0}sec)"
  end
end
open(LOGFILE, "a") {|f| f.puts "==== end (#{Time.now})" }


