#!/usr/bin/ruby

require 'zlib'
require "neural.rb"

N_IMAGES   = if ARGV[0] then ARGV[0].to_i else 2 end
N_LEARNING = if ARGV[1] then ARGV[1].to_i else 2 end
N_TRIALS   = if ARGV[2] then ARGV[2].to_i else 2 end
LOGFILE = "mnist2.log"

# load training data
def load_mnist(image_file, label_file)
  n_rows = n_cols = nil
  images = []
  labels = []
  Zlib::GzipReader.open(image_file) do |f|
    magic, n_images = f.read(8).unpack('N2')
    raise 'This is not MNIST image file' if magic != 2051
    n_rows, n_cols = f.read(8).unpack('N2')
    n_images.times do
      images << f.read(n_rows * n_cols)
    end
  end

  Zlib::GzipReader.open(label_file) do |f|
    magic, n_labels = f.read(8).unpack('N2')
    raise 'This is not MNIST label file' if magic != 2049
    labels = f.read(n_labels).unpack('C*')
  end
  [images, labels]
end

images, labels = load_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
test_images, test_labels = load_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

# network
def generate_network(network_type)
  in_units = (1..(28*28)).map{|i| Unit.new("x#{i}")}
  convolutions1 = (1..(24*24)).map{|i| IdentityUnit.new("c#{i}")}
  convolutions2 = (1..(24*24)).map{|i| IdentityUnit.new("d#{i}")}
  convolutions3 = (1..(24*24)).map{|i| IdentityUnit.new("e#{i}")}
  hiddenunits1 = (1..100).map{|i| TanhUnit.new("z#{i}")}
  hiddenunits2 = (1..30).map{|i| TanhUnit.new("w#{i}")}
  out_unit = (1..10).map{|i| SoftMaxUnit.new("y#{i}")}

  # network
  network = Network.new(:error_func=>ErrorFunction::SoftMax)
  network.in = in_units

  name = nil
  case network_type
  when 0
    name = "30 hiddens"
    network.link in_units, hiddenunits2
    network.link hiddenunits2, out_unit
  when 1
    name = "100 hiddens"
    network.link in_units, hiddenunits1
    network.link hiddenunits1, out_unit
  when 2
    name = "100 hiddens + 60 hiddens"
    network.link in_units, hiddenunits1
    network.link hiddenunits1, hiddenunits2
    network.link hiddenunits2, out_unit
  when 3
    name = "1 convolution + 60 hiddens"
    network.convolutional_link in_units, convolutions1, 28, 28, 5
    network.link convolutions1, hiddenunits2
    network.link hiddenunits2, out_unit
  when 4
    name = "1 convolution + 100 hiddens"
    network.convolutional_link in_units, convolutions1, 28, 28, 5
    network.link convolutions1, hiddenunits1
    network.link hiddenunits1, out_unit
  when 5
    name = "2 convolution + 60 hiddens"
    network.convolutional_link in_units, convolutions1, 28, 28, 5
    network.convolutional_link in_units, convolutions2, 28, 28, 5
    network.link convolutions1+convolutions2, hiddenunits2
    network.link hiddenunits2, out_unit
  when 6
    name = "2 convolution + 100 hiddens"
    network.convolutional_link in_units, convolutions1, 28, 28, 5
    network.convolutional_link in_units, convolutions2, 28, 28, 5
    network.link convolutions1+convolutions2, hiddenunits1
    network.link hiddenunits1, out_unit
  when 7
    name = "3 convolution + 60 hiddens"
    network.convolutional_link in_units, convolutions1, 28, 28, 5
    network.convolutional_link in_units, convolutions2, 28, 28, 5
    network.convolutional_link in_units, convolutions3, 28, 28, 5
    network.link convolutions1+convolutions2+convolutions3, hiddenunits2
    network.link hiddenunits2, out_unit
  when 8
    name = "3 convolution + 100 hiddens"
    network.convolutional_link in_units, convolutions1, 28, 28, 5
    network.convolutional_link in_units, convolutions2, 28, 28, 5
    network.convolutional_link in_units, convolutions3, 28, 28, 5
    network.link convolutions1+convolutions2+convolutions3, hiddenunits1
    network.link hiddenunits1, out_unit
  when 9
    name = "3 convolution + 100 hiddens + 60 hiddens"
    network.convolutional_link in_units, convolutions1, 28, 28, 5
    network.convolutional_link in_units, convolutions2, 28, 28, 5
    network.convolutional_link in_units, convolutions3, 28, 28, 5
    network.link convolutions1+convolutions2+convolutions3, hiddenunits1
    network.link hiddenunits1, hiddenunits2
    network.link hiddenunits2, out_unit
  end

  network.out = out_unit
  [name, network]
end


# start
open(LOGFILE, "a") {|f| f.puts "==== start (#{Time.now}), N_IMAGES=#{N_IMAGES}, N_LEARNING=#{N_LEARNING}, N_TRIALS=#{N_TRIALS}" }
10.times do |network_type|

  name, network = generate_network(network_type)
  amount_correct = amount_mistake = 0
  open(LOGFILE, "a") {|f| f.puts "---- #{name}" }

  t1 = Time.now.to_i

  N_TRIALS.times do |trial|
    network.weights.init_parameters

    # training
    N_LEARNING.times do |n|
      # select training data
      map = Hash.new
      map[rand(images.length)] = 1 while map.size < N_IMAGES
      training_index = map.keys

      eta = if n<2 then 0.1 elsif n<5 then 0.05 else 0.01 end
      training_index.sort_by{rand}.each do |idx|
        image = images[idx].unpack('C*')
        target = [0]*10
        target[labels[idx]] = 1

        grad = Gradient::BackPropagate.call(network, image, target)
        network.weights.descent eta, grad
      end
    end

  open(LOGFILE, "a") {|f| f.puts Time.now.to_i-t1 }

    # test
    correct = mistake = 0
    test_images.each_with_index do |image_binary, idx|
      image = image_binary.unpack('C*')
      y = network.apply(*image)
      predict = (0..9).max_by{|i| y[i]}
      if test_labels[idx] == predict
        correct += 1
      else
        mistake += 1
      end
    end

    open(LOGFILE, "a") do |f|
      f.puts "trial #{trial+1}: p=#{correct}, n=#{mistake}, rate=#{(correct.to_f/(correct+mistake)*10000).floor/100.0}%, time=#{Time.now.to_i-t1}s"
    end

    amount_correct += correct
    amount_mistake += mistake
  end

  t2 = Time.now.to_i
  open(LOGFILE, "a") do |f|
    f.puts "result of '#{name}': p=#{amount_correct}, n=#{amount_mistake}, rate=#{(amount_correct.to_f/(amount_correct+amount_mistake)*10000).floor/100.0}%, # of w=#{network.weights.parameters.length}, time=#{t2-t1}s"
  end
end
open(LOGFILE, "a") {|f| f.puts "==== end (#{Time.now})" }

