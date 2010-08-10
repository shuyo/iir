#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

require 'optparse'
require 'nkf'
require 'detect.rb'

parser = OptionParser.new
model = 'model'
alpha = 1.0
debug_flag = false
parser.on('-f VAL', String, 'model filename') {|v| model = v }
parser.on('-a VAL', Float, 'alpha (additive smoothing)') {|v| alpha = v }
parser.on('-d', 'debug mode') { debug_flag = true }
parser.parse!(ARGV)

detector = LanguageDetector::Detector.new(model)
detector.debug_on if debug_flag

ARGV.each do |filename|
  text = open(filename){|f| NKF.nkf('-w', f.read) }
  text.gsub!(/https?:\/\/[0-9a-zA-Z\.\/\?=\&\-]+/, '')
  problist = detector.detect(text, alpha)
  puts "#{filename},#{problist.inspect},#{text[0..100].gsub(/\s+/, ' ').strip}"
end
