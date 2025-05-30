//
// Autogenerated by Thrift Compiler (0.9.0)
//
// DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
//

Position = function(args) {
  this.ticker = null;
  this.price = null;
  this.shares = null;
  if (args) {
    if (args.ticker !== undefined) {
      this.ticker = args.ticker;
    }
    if (args.price !== undefined) {
      this.price = args.price;
    }
    if (args.shares !== undefined) {
      this.shares = args.shares;
    }
  }
};
Position.prototype = {};
Position.prototype.read = function(input) {
  input.readStructBegin();
  while (true)
  {
    var ret = input.readFieldBegin();
    var fname = ret.fname;
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
      if (ftype == Thrift.Type.STRING) {
        this.ticker = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 2:
      if (ftype == Thrift.Type.DOUBLE) {
        this.price = input.readDouble().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 3:
      if (ftype == Thrift.Type.I64) {
        this.shares = input.readI64().value;
      } else {
        input.skip(ftype);
      }
      break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Position.prototype.write = function(output) {
  output.writeStructBegin('Position');
  if (this.ticker !== null && this.ticker !== undefined) {
    output.writeFieldBegin('ticker', Thrift.Type.STRING, 1);
    output.writeString(this.ticker);
    output.writeFieldEnd();
  }
  if (this.price !== null && this.price !== undefined) {
    output.writeFieldBegin('price', Thrift.Type.DOUBLE, 2);
    output.writeDouble(this.price);
    output.writeFieldEnd();
  }
  if (this.shares !== null && this.shares !== undefined) {
    output.writeFieldBegin('shares', Thrift.Type.I64, 3);
    output.writeI64(this.shares);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

Portfolio = function(args) {
  this.name = null;
  this.constituents = null;
  this.basis = null;
  this.price = null;
  this.largest_10day_loss = null;
  this.largest_10day_loss_date = null;
  this.hist_prices = null;
  if (args) {
    if (args.name !== undefined) {
      this.name = args.name;
    }
    if (args.constituents !== undefined) {
      this.constituents = args.constituents;
    }
    if (args.basis !== undefined) {
      this.basis = args.basis;
    }
    if (args.price !== undefined) {
      this.price = args.price;
    }
    if (args.largest_10day_loss !== undefined) {
      this.largest_10day_loss = args.largest_10day_loss;
    }
    if (args.largest_10day_loss_date !== undefined) {
      this.largest_10day_loss_date = args.largest_10day_loss_date;
    }
    if (args.hist_prices !== undefined) {
      this.hist_prices = args.hist_prices;
    }
  }
};
Portfolio.prototype = {};
Portfolio.prototype.read = function(input) {
  input.readStructBegin();
  while (true)
  {
    var ret = input.readFieldBegin();
    var fname = ret.fname;
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
      if (ftype == Thrift.Type.STRING) {
        this.name = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 2:
      if (ftype == Thrift.Type.LIST) {
        var _size0 = 0;
        var _rtmp34;
        this.constituents = [];
        var _etype3 = 0;
        _rtmp34 = input.readListBegin();
        _etype3 = _rtmp34.etype;
        _size0 = _rtmp34.size;
        for (var _i5 = 0; _i5 < _size0; ++_i5)
        {
          var elem6 = null;
          elem6 = new Position();
          elem6.read(input);
          this.constituents.push(elem6);
        }
        input.readListEnd();
      } else {
        input.skip(ftype);
      }
      break;
      case 3:
      if (ftype == Thrift.Type.DOUBLE) {
        this.basis = input.readDouble().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 4:
      if (ftype == Thrift.Type.DOUBLE) {
        this.price = input.readDouble().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 5:
      if (ftype == Thrift.Type.DOUBLE) {
        this.largest_10day_loss = input.readDouble().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 6:
      if (ftype == Thrift.Type.STRING) {
        this.largest_10day_loss_date = input.readString().value;
      } else {
        input.skip(ftype);
      }
      break;
      case 7:
      if (ftype == Thrift.Type.LIST) {
        var _size7 = 0;
        var _rtmp311;
        this.hist_prices = [];
        var _etype10 = 0;
        _rtmp311 = input.readListBegin();
        _etype10 = _rtmp311.etype;
        _size7 = _rtmp311.size;
        for (var _i12 = 0; _i12 < _size7; ++_i12)
        {
          var elem13 = null;
          elem13 = input.readDouble().value;
          this.hist_prices.push(elem13);
        }
        input.readListEnd();
      } else {
        input.skip(ftype);
      }
      break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Portfolio.prototype.write = function(output) {
  output.writeStructBegin('Portfolio');
  if (this.name !== null && this.name !== undefined) {
    output.writeFieldBegin('name', Thrift.Type.STRING, 1);
    output.writeString(this.name);
    output.writeFieldEnd();
  }
  if (this.constituents !== null && this.constituents !== undefined) {
    output.writeFieldBegin('constituents', Thrift.Type.LIST, 2);
    output.writeListBegin(Thrift.Type.STRUCT, this.constituents.length);
    for (var iter14 in this.constituents)
    {
      if (this.constituents.hasOwnProperty(iter14))
      {
        iter14 = this.constituents[iter14];
        iter14.write(output);
      }
    }
    output.writeListEnd();
    output.writeFieldEnd();
  }
  if (this.basis !== null && this.basis !== undefined) {
    output.writeFieldBegin('basis', Thrift.Type.DOUBLE, 3);
    output.writeDouble(this.basis);
    output.writeFieldEnd();
  }
  if (this.price !== null && this.price !== undefined) {
    output.writeFieldBegin('price', Thrift.Type.DOUBLE, 4);
    output.writeDouble(this.price);
    output.writeFieldEnd();
  }
  if (this.largest_10day_loss !== null && this.largest_10day_loss !== undefined) {
    output.writeFieldBegin('largest_10day_loss', Thrift.Type.DOUBLE, 5);
    output.writeDouble(this.largest_10day_loss);
    output.writeFieldEnd();
  }
  if (this.largest_10day_loss_date !== null && this.largest_10day_loss_date !== undefined) {
    output.writeFieldBegin('largest_10day_loss_date', Thrift.Type.STRING, 6);
    output.writeString(this.largest_10day_loss_date);
    output.writeFieldEnd();
  }
  if (this.hist_prices !== null && this.hist_prices !== undefined) {
    output.writeFieldBegin('hist_prices', Thrift.Type.LIST, 7);
    output.writeListBegin(Thrift.Type.DOUBLE, this.hist_prices.length);
    for (var iter15 in this.hist_prices)
    {
      if (this.hist_prices.hasOwnProperty(iter15))
      {
        iter15 = this.hist_prices[iter15];
        output.writeDouble(iter15);
      }
    }
    output.writeListEnd();
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

