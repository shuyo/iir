#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

require 'mysql'
require 'json'
require 'common.rb'
require 'detect.rb'

parser= LD::optparser
opt = {:N=>3, :training_size=>150, :csv=>false, :json=>false}
parser.on('-n VAL', Integer, 'N-gram') {|v| opt[:N] = v }
parser.on('--size=VAL', Integer, 'max size of training data') {|v| opt[:training_size] = v }
parser.on('--csv') {|v| opt[:csv] = true }
parser.parse!(ARGV)

# Database
db = LD::db_connect
#ps_select = db.prepare("select title,lang,body from news order by id desc")
ps_select = db.prepare("select title,lang,body from news order by rand()")

ps_select.execute
n_k = Hash.new(0)
p_ik = Hash.new{|h,k| h[k]=Hash.new(0)}
ngramer = LanguageDetector::Ngramer.new(opt[:N])
while rs = ps_select.fetch
  title, lang, body = rs
  title.sub!(/ - [^\-]+$/, '')
  next if n_k[lang] >= opt[:training_size]
  n_k[lang] += 1
  text = LD::decode_entity(title + "\n" + body)

  grams = Hash.new
  ngramer.clear
  text.scan(/./) do |x|
    ngramer.append x
    ngramer.each do |z|
      grams[z] = 1
    end
  end
  grams.each do |gram, dummy|
    p_ik[gram][lang] += 1
  end
end

if opt[:csv]
  puts ","+LD::LANGLIST.join(',')
  p_ik.to_a.sort.each do |unigram,langs|
    langs.default = ''
    puts "'#{unigram.unpack('H*')[0]},#{LD::LANGLIST.map{|lang| langs[lang]}.join(',')}"
  end
end

keys = p_ik.keys
keys.each do |chunk|
  langs = p_ik[chunk].keys
  langs.each do |lang|
    p_ik[chunk].delete lang if p_ik[chunk][lang] <= 2
  end
  p_ik.delete chunk if p_ik[chunk].size == 0
end

p_ik.default = 0
open(LD::model_filename, 'w') do |f|
  JSON.dump([n_k, p_ik, opt[:N]], f)
end

