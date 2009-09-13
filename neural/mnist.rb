#!/usr/bin/ruby

require 'zlib'
require "neural.rb"

n_rows = n_cols = nil
images = []
labels = []
Zlib::GzipReader.open('train-images-idx3-ubyte.gz') do |f|
  magic, n_images = f.read(8).unpack('N2')
  raise 'This is not MNIST image file' if magic != 2051
  n_rows, n_cols = f.read(8).unpack('N2')
  n_images.times do
    images << f.read(n_rows * n_cols)
  end
end

Zlib::GzipReader.open('train-labels-idx1-ubyte.gz') do |f|
  magic, n_labels = f.read(8).unpack('N2')
  raise 'This is not MNIST label file' if magic != 2049
  labels = f.read(n_labels).unpack('C*')
end

# output pgm
def output_pgm(filename, images, n_rows, n_cols, n_width, n_height)
  open(filename, "wb") do |f|
    f.printf("P5 %d %d %d ", n_rows*n_width, n_cols*n_height, 0xff)
    offset = 0
    buf = ""
    n_height.times do
      n_cols.times do |y|
        n_width.times do |idx|
          st = images[offset + idx][y * n_rows, n_rows].unpack('C*').map{|p| 0xff - p }.pack("C*")
          #p images[offset + idx][y * n_rows, n_rows].unpack('C*') if st.length != 28
          buf << st
        end
      end
      offset += n_width
    end
    f.puts buf
  end
end
#output_pgm "mnist.pgm", images, n_rows, n_cols, 300, 200

# units
in_units = (1..(28*28)).map{|i| Unit.new("x#{i}")}
hiddenunits = (1..100).map{|i| TanhUnit.new("z#{i}")}
out_unit = (1..10).map{|i| SoftMaxUnit.new("y#{i}")}

# network
network = Network.new(:error_func=>ErrorFunction::SoftMax, :code_generate=>true)
network.in  = in_units
network.link in_units, hiddenunits
network.link hiddenunits, out_unit
network.out = out_unit

# training
t1 = Time.now.to_f
N_IMAGES = 1000
10.times do |n|
  eta = if n<2 then 0.1 elsif n<5 then 0.05 else 0.01 end
  (0..(N_IMAGES-1)).sort_by{rand}.each do |idx|
    image = images[idx].unpack('C*')
    target = [0]*10
    target[labels[idx]] = 1

    puts "(#{n+1}, #{idx}): correct: #{labels[idx]}"
    #puts "#{network.apply(*image).map{|x| (x*10000).floor/10000.0}.inspect}, e=#{(network.error_function(image, target)*1000)/1000.0}"

    grad = Gradient::BackPropagate.call(network, image, target)
    network.weights.descent eta, grad

    #puts "#{network.apply(*image).map{|x| (x*10000).floor/10000.0}.inspect}, e=#{(network.error_function(image, target)*1000)/1000.0}"
  end
end
t2 = Time.now.to_f

# test
puts "------------------------------"
correct = mistake = 0
(0..(N_IMAGES*2-1)).each do |idx|
  image = images[idx].unpack('C*')
  target = [0]*10
  target[labels[idx]] = 1

  y = network.apply(*image)
  predict = (0..9).max_by{|i| y[i]}
  puts "#{idx}: predict = #{predict}, expect = #{labels[idx]}"
  puts "#{y.map{|x| (x*10000).floor/10000.0}.inspect}, e=#{(network.error_function(image, target)*1000)/1000.0}"
  if labels[idx] == predict
    correct += 1
  else
    mistake += 1
  end
end
t3 = Time.now.to_f

#
puts "correct = #{correct}, mistake = #{mistake}, rate = #{(correct.to_f/(correct+mistake)*10000).floor/100.0}%"

puts "learning: #{t2-t1}"
puts "testing: #{t3-t2}"


