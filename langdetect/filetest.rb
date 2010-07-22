#!/usr/bin/ruby -Ku

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

detector = LanguageDetector::Detector.new(model, alpha)
detector.debug_on if debug_flag

ngramer = detector.ngramer
ARGV.each do |filename|
  text = open(filename){|f| NKF.nkf('-w', f.read) }
  text.gsub!(/https?:\/\/[0-9a-zA-Z\.\/\?=\&\-]+/, '')

  rate = text.length / 100.0
  rate = 1 if rate > 1
  detector.init alpha * rate
  ngramer.clear
  text.scan(/./) do |x|
    ngramer.append x
    ngramer.each do |z|
      detector.append z
    end
    break if detector.maxprob > 0.99999
  end

  problist = detector.problist
  puts "#{filename},#{problist.inspect},#{text[0..100].gsub(/\s+/, ' ').strip}"
end
