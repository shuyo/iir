#!/usr/bin/ruby -Ku

require 'mysql'
require 'optparse'

module LD
  ENTITIES = {
    "&#39;"=>"'",
    "&amp;"=>"&",
    "&gt;"=>">",
    "&lt;"=>"<",
    "&quot;"=>'"',
    "&raquo;"=>""
  }

  def self.optparser
    @opt = {
      :host=>'localhost', :user=>'root', :passwd=>'', :dbname=>'googlenews', :port=>3306,
      :model=>'model.json'
    }

    parser = OptionParser.new
    parser.on('--host=VAL', String, 'database host') {|v| @opt[:host] = v }
    parser.on('--user=VAL', String, 'database user') {|v| @opt[:user] = v }
    parser.on('--password=VAL', String, 'database password') {|v| @opt[:passwd] = v }
    parser.on('--dbname=VAL', String, 'database name') {|v| @opt[:dbname] = v }
    parser.on('--port=VAL', Integer, 'database port') {|v| @opt[:port] = v }
    parser.on('-f VAL', String, 'model filename') {|v| @opt[:model] = v }
    parser
  end
  def self.model_filename
    @opt[:model]
  end

  def self.db_connect
    db = Mysql::init
    db.options Mysql::SET_CHARSET_NAME, 'utf8'
    db.real_connect @opt[:host], @opt[:user], @opt[:passwd], @opt[:dbname], @opt[:port]
    db
  end

  def self.decode_entity(st)
    st.gsub(/&[^ &]+?;/){|m| ENTITIES[m] || m}
  end
end

